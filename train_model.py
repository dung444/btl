"""Huấn luyện hệ thống gợi ý thời trang từ dữ liệu Kaggle.

Mô hình kết hợp tên sản phẩm, thuộc tính phân loại, năm và màu trong ảnh.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize


# Khai báo đường dẫn tập dữ liệu và nơi lưu mô hình để chạy lại dễ dàng.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "fashion"
CSV_PATH = DATA_DIR / "styles.csv"
IMAGE_DIR = DATA_DIR / "images"
MODEL_DIR = ROOT / "models"

CATEGORICAL_COLUMNS = [
    "gender", "masterCategory", "subCategory", "baseColour", "season", "usage"
]


def load_products(limit: int | None = 15000) -> tuple[pd.DataFrame, int]:
    """Đọc dữ liệu, làm sạch thuộc tính và chỉ giữ sản phẩm có ảnh thật."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {CSV_PATH}")

    # Một số dòng CSV có lỗi dấu phân cách; bỏ qua đúng các dòng hỏng đó.
    data = pd.read_csv(CSV_PATH, on_bad_lines="skip", low_memory=False)
    data = data.dropna(subset=["id", "productDisplayName"]).drop_duplicates("id")
    data["id"] = pd.to_numeric(data["id"], errors="coerce")
    data = data.dropna(subset=["id"]).copy()
    data["id"] = data["id"].astype(int)
    # Quét thư mục ảnh một lần, tránh hàng chục nghìn lần kiểm tra đường dẫn riêng lẻ.
    available_images = {int(path.stem) for path in IMAGE_DIR.glob("*.jpg") if path.stem.isdigit()}
    data = data[data["id"].isin(available_images)].copy()
    available_count = len(data)
    if limit is not None and len(data) > limit:
        # Lấy mẫu tái lập được và có đại diện từ từng nhóm hàng.
        data = data.sample(n=limit, random_state=42)
    data["image_path"] = data["id"].map(lambda value: str(IMAGE_DIR / f"{value}.jpg"))

    # Điền giá trị thiếu để bộ mã hóa không gặp lỗi với trường phân loại.
    for column in CATEGORICAL_COLUMNS + ["articleType"]:
        data[column] = data[column].fillna("Không rõ").astype(str).str.strip()
    data["productDisplayName"] = data["productDisplayName"].astype(str).str.strip()
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["year"] = data["year"].fillna(data["year"].median())
    return data.reset_index(drop=True), available_count


def image_features(image_path: str) -> np.ndarray:
    """Lấy trung bình RGB và histogram màu từ ảnh sản phẩm."""
    try:
        with Image.open(image_path) as original:
            pixels = np.asarray(original.convert("RGB").resize((24, 24)), dtype=np.float32) / 255.0
        means = pixels.mean(axis=(0, 1))
        histogram = np.concatenate([
            np.histogram(pixels[:, :, channel], bins=8, range=(0, 1))[0] / pixels.shape[0] ** 2
            for channel in range(3)
        ])
        return np.concatenate([means, histogram]).astype(np.float32)
    except (OSError, ValueError):
        # Ảnh bị hỏng được biểu diễn bằng vector 0, các thuộc tính khác vẫn dùng được.
        return np.zeros(27, dtype=np.float32)


def encode_features(fit_data: pd.DataFrame, query_data: pd.DataFrame,
                    fit_colours: np.ndarray, query_colours: np.ndarray):
    """Học bộ biến đổi trên tập huấn luyện rồi mã hóa cả tập truy vấn."""
    text_encoder = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=2,
        max_features=12000, sublinear_tf=True,
    )
    text_train = text_encoder.fit_transform(fit_data["productDisplayName"])
    text_query = text_encoder.transform(query_data["productDisplayName"])

    # articleType chỉ làm nhãn kiểm tra, không nằm trong nhóm đặc trưng đầu vào.
    category_encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    category_train = category_encoder.fit_transform(fit_data[CATEGORICAL_COLUMNS])
    category_query = category_encoder.transform(query_data[CATEGORICAL_COLUMNS])

    year_encoder = StandardScaler()
    year_train = np.clip(year_encoder.fit_transform(fit_data[["year"]]), -3, 3)
    year_query = np.clip(year_encoder.transform(query_data[["year"]]), -3, 3)
    colour_train = normalize(csr_matrix(fit_colours), norm="l2")
    colour_query = normalize(csr_matrix(query_colours), norm="l2")

    # Ma trận thứ nhất chỉ có văn bản, thứ hai chỉ có thuộc tính cấu trúc.
    text_only_train = normalize(text_train)
    text_only_query = normalize(text_query)
    structured_train = normalize(hstack([
        category_train * 1.25, csr_matrix(year_train) * 0.15,
    ], format="csr"))
    structured_query = normalize(hstack([
        category_query * 1.25, csr_matrix(year_query) * 0.15,
    ], format="csr"))

    # Mô hình kết hợp thêm đặc trưng màu từ ảnh thật.
    hybrid_train = normalize(hstack([
        text_train * 2.0, category_train * 1.25,
        csr_matrix(year_train) * 0.15, colour_train * 0.8,
    ], format="csr", dtype=np.float32))
    hybrid_query = normalize(hstack([
        text_query * 2.0, category_query * 1.25,
        csr_matrix(year_query) * 0.15, colour_query * 0.8,
    ], format="csr", dtype=np.float32))

    matrices = {
        "Chỉ tên": (text_only_train, text_only_query),
        "Chỉ thuộc tính": (structured_train, structured_query),
        "Kết hợp": (hybrid_train, hybrid_query),
    }
    encoders = {
        "text_encoder": text_encoder, "category_encoder": category_encoder,
        "year_encoder": year_encoder,
    }
    return matrices, encoders


def evaluate_retrieval(train_data: pd.DataFrame, query_data: pd.DataFrame,
                       train_matrix: csr_matrix, query_matrix: csr_matrix) -> dict:
    """Đánh giá Top 10 trên sản phẩm chưa có trong tập huấn luyện."""
    # Chỉ lấy tối đa 200 truy vấn để thời gian đánh giá ổn định trên máy cá nhân.
    rng = np.random.default_rng(42)
    sample_indexes = rng.choice(len(query_data), size=min(200, len(query_data)), replace=False)
    reference_types = train_data["articleType"].to_numpy()
    query_types = query_data.iloc[sample_indexes]["articleType"].to_numpy()
    precision_scores, hit_scores = [], []
    for matrix_row, true_type in zip(sample_indexes, query_types):
        scores = (query_matrix[matrix_row] @ train_matrix.T).toarray().ravel()
        top_indexes = np.argsort(-scores)[:10]
        same_type = reference_types[top_indexes] == true_type
        precision_scores.append(float(np.mean(same_type)))
        hit_scores.append(float(np.any(same_type)))
    return {
        "precision_at_10": round(float(np.mean(precision_scores)), 4),
        "hit_rate_at_10": round(float(np.mean(hit_scores)), 4),
        "queries": len(sample_indexes),
    }


def train(limit: int | None = 15000) -> dict:
    """So sánh ba mô hình trên tập kiểm tra và lưu mô hình tốt để sử dụng."""
    products, available_count = load_products(limit)
    if len(products) < 20:
        raise ValueError("Dataset cần tối thiểu 20 sản phẩm có ảnh để huấn luyện.")

    # Trích xuất ảnh một lần, dùng lại cho kiểm tra và mô hình phục vụ giao diện.
    colour_features = np.stack([image_features(path) for path in products["image_path"]])
    indexes = np.arange(len(products))
    train_indexes, test_indexes = train_test_split(
        indexes, test_size=0.2, random_state=42,
        stratify=products["masterCategory"],
    )
    train_data = products.iloc[train_indexes]
    test_data = products.iloc[test_indexes]
    matrices, _ = encode_features(
        train_data, test_data,
        colour_features[train_indexes], colour_features[test_indexes],
    )
    comparison = {
        name: evaluate_retrieval(train_data, test_data, fit_matrix, query_matrix)
        for name, (fit_matrix, query_matrix) in matrices.items()
    }

    # Sau khi đánh giá, học lại trên toàn bộ dữ liệu để giao diện có nhiều gợi ý hơn.
    full_matrices, encoders = encode_features(
        products, products, colour_features, colour_features,
    )
    full_matrix = full_matrices["Kết hợp"][0]
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    model.fit(full_matrix)
    metrics = {
        "products": int(len(products)),
        "dataset_products": int(available_count),
        "train_products": int(len(train_indexes)),
        "test_products": int(len(test_indexes)),
        "main_categories": int(products["masterCategory"].nunique()),
        "sub_categories": int(products["subCategory"].nunique()),
        "article_types": int(products["articleType"].nunique()),
        "feature_count": int(full_matrix.shape[1]),
        "comparison": comparison,
        "evaluation_note": "articleType không nằm trong đặc trưng; truy vấn kiểm tra không có trong tập huấn luyện.",
    }
    MODEL_DIR.mkdir(exist_ok=True)
    products.to_pickle(MODEL_DIR / "products.pkl")
    joblib.dump({"model": model, "matrix": full_matrix, **encoders},
                MODEL_DIR / "fashion_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(comparison).T.to_csv(MODEL_DIR / "model_comparison.csv", index_label="Mô hình")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình gợi ý thời trang")
    parser.add_argument("--full", action="store_true", help="Huấn luyện trên toàn bộ sản phẩm")
    args = parser.parse_args()
    # Dùng chuỗi ASCII khi in ra terminal Windows để tránh lỗi bảng mã cục bộ.
    print(json.dumps(train(limit=None if args.full else 15000), ensure_ascii=True, indent=2))
