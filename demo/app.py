"""Giao diện tiếng Việt cho ứng dụng gợi ý sản phẩm thời trang."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from catalog_search import search_products


# Tải các tệp đã huấn luyện một lần để thao tác trên trang diễn ra nhanh.
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
st.set_page_config(page_title="Gợi ý sản phẩm thời trang", page_icon="👗", layout="wide")
st.markdown("""
<style>
.block-container{max-width:1350px;padding-top:1.4rem}
.hero{background:linear-gradient(125deg,#182136,#3b557d);color:white;border-radius:18px;padding:1.6rem 2rem;margin-bottom:1rem}
.hero h1{color:white;margin:0}.hero p{margin:.4rem 0 0;color:#e4ebf4}
[data-testid="stMetric"]{background:#f6f8fc;border:1px solid #e6eaf0;border-radius:12px;padding:.8rem}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(file_versions: tuple[int, int, int]):
    """Đọc sản phẩm, ma trận đặc trưng và kết quả huấn luyện từ ổ đĩa."""
    products = pd.read_pickle(MODEL_DIR / "products.pkl")
    bundle = joblib.load(MODEL_DIR / "fashion_model.joblib")
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    return products, bundle["matrix"], metrics


if not (MODEL_DIR / "fashion_model.joblib").exists():
    st.error("Chưa có mô hình. Hãy mở và chạy file `app/ML_product_recommender.ipynb` rồi tải lại trang.")
    st.stop()

file_versions = tuple((MODEL_DIR / filename).stat().st_mtime_ns for filename in [
    "products.pkl", "fashion_model.joblib", "metrics.json"
])
products, matrix, metrics = load_model(file_versions)
if not {"train_products", "test_products", "comparison"}.issubset(metrics):
    st.error("Bộ kết quả đánh giá chưa khớp với giao diện. Hãy mở và chạy lại file `app/ML_product_recommender.ipynb` rồi tải lại trang.")
    st.stop()
st.markdown('<div class="hero"><h1>👗 Hệ thống gợi ý sản phẩm thời trang</h1><p>Khám phá sản phẩm tương tự theo tên, danh mục, màu, mùa, mục đích sử dụng và ảnh thật.</p></div>', unsafe_allow_html=True)

# Bộ lọc nằm ngoài các tab để áp dụng ngay khi người dùng thay đổi.
with st.sidebar:
    st.header("Bộ lọc sản phẩm")
    department = st.selectbox("Nhóm hàng", ["Tất cả"] + sorted(products["masterCategory"].unique().tolist()))
    subcategory_options = products if department == "Tất cả" else products[products["masterCategory"] == department]
    subcategory = st.selectbox("Danh mục phụ", ["Tất cả"] + sorted(subcategory_options["subCategory"].unique().tolist()))
    article_options = subcategory_options if subcategory == "Tất cả" else subcategory_options[subcategory_options["subCategory"] == subcategory]
    article_type = st.selectbox("Loại sản phẩm", ["Tất cả"] + sorted(article_options["articleType"].unique().tolist()))
    gender = st.selectbox("Đối tượng", ["Tất cả"] + sorted(products["gender"].unique().tolist()))
    colour = st.selectbox("Màu sắc", ["Tất cả"] + sorted(products["baseColour"].unique().tolist()))
    season = st.selectbox("Mùa", ["Tất cả"] + sorted(products["season"].unique().tolist()))
    usage = st.selectbox("Mục đích sử dụng", ["Tất cả"] + sorted(products["usage"].unique().tolist()))
    minimum_year = int(products["year"].min())
    maximum_year = int(products["year"].max())
    year_range = st.slider("Khoảng năm", minimum_year, maximum_year, (minimum_year, maximum_year))
    top_n = st.slider("Số sản phẩm gợi ý", 4, 20, 12)
    st.caption("Dữ liệu ảnh và thuộc tính: Fashion Product Images (Small) từ Kaggle.")

tab_recommend, tab_explore, tab_model = st.tabs(["Gợi ý sản phẩm", "Khám phá dữ liệu", "Mô hình học máy"])

with tab_recommend:
    st.subheader("Tìm sản phẩm bạn thích")
    keyword = st.text_input("Tìm theo tên hoặc thuộc tính", placeholder="Ví dụ: váy đỏ, áo sơ mi, giày, túi...")
    pool = search_products(products, keyword)
    if department != "Tất cả":
        pool = pool[pool["masterCategory"] == department]
    if gender != "Tất cả":
        pool = pool[pool["gender"] == gender]
    if subcategory != "Tất cả":
        pool = pool[pool["subCategory"] == subcategory]
    if article_type != "Tất cả":
        pool = pool[pool["articleType"] == article_type]

    if pool.empty:
        st.info("Không tìm thấy sản phẩm phù hợp. Hãy thử từ khóa khác hoặc nới lỏng bộ lọc nhóm hàng/loại sản phẩm.")
    else:
        st.caption(f"Tìm thấy {len(pool):,} sản phẩm phù hợp để chọn làm mẫu.")
        # Chỉ đưa một phần danh sách vào ô chọn để trình duyệt phản hồi nhanh.
        choices = pool.index.tolist()[:600]
        selected = st.multiselect(
            "Chọn tối đa 3 sản phẩm mẫu", choices, default=choices[:1], max_selections=3,
            format_func=lambda index: f"{products.at[index, 'productDisplayName']} (#{products.at[index, 'id']})",
        )
        if selected:
            # Xem sản phẩm mẫu trước khi nhận kết quả để biết mô hình đang dựa trên gì.
            with st.expander("Xem sản phẩm đã chọn", expanded=False):
                for index in selected:
                    reference = products.loc[index]
                    left, right = st.columns([1, 3])
                    with left:
                        st.image(reference["image_path"], width="stretch")
                    with right:
                        st.markdown(f"**{reference['productDisplayName']}**")
                        st.write(f"Loại: {reference['articleType']} · Danh mục: {reference['subCategory']}")
                        st.write(f"Màu: {reference['baseColour']} · Mùa: {reference['season']} · Mục đích: {reference['usage']}")

            # Trung bình các sản phẩm được chọn thành hồ sơ sở thích tạm thời.
            profile = normalize(csr_matrix(np.asarray(matrix[selected].mean(axis=0))))
            similarity = (profile @ matrix.T).toarray().ravel()
            similarity[selected] = -1
            candidates = products.copy()
            candidates["score"] = similarity
            if department != "Tất cả": candidates = candidates[candidates["masterCategory"] == department]
            if subcategory != "Tất cả": candidates = candidates[candidates["subCategory"] == subcategory]
            if article_type != "Tất cả": candidates = candidates[candidates["articleType"] == article_type]
            if gender != "Tất cả": candidates = candidates[candidates["gender"] == gender]
            if colour != "Tất cả": candidates = candidates[candidates["baseColour"] == colour]
            if season != "Tất cả": candidates = candidates[candidates["season"] == season]
            if usage != "Tất cả": candidates = candidates[candidates["usage"] == usage]
            candidates = candidates[candidates["year"].between(*year_range)]
            candidates = candidates[candidates["score"] >= 0].nlargest(top_n, "score")
            if candidates.empty:
                st.info("Không có sản phẩm phù hợp đồng thời với tất cả bộ lọc.")
            else:
                st.caption(f"{len(candidates)} sản phẩm được xếp theo độ tương tự với lựa chọn của bạn.")
                references = products.loc[selected]
                for start in range(0, len(candidates), 4):
                    columns = st.columns(4)
                    for column, (_, product) in zip(columns, candidates.iloc[start:start + 4].iterrows()):
                        with column:
                            # Ảnh được đọc trực tiếp từ dataset đã tải; không dùng ảnh minh họa.
                            st.image(product["image_path"], width="stretch")
                            st.markdown(f"**{product['productDisplayName']}**")
                            st.caption(f"{product['articleType']} · {product['baseColour']} · {product['season']} · {product['usage']}")
                            reasons = [label for field, label in [
                                ("subCategory", "cùng danh mục"), ("baseColour", "cùng màu"),
                                ("season", "cùng mùa"), ("usage", "cùng mục đích"),
                            ] if product[field] in references[field].values]
                            st.caption("Điểm tương tự: " + f"{product['score']:.2f}" + (" · " + ", ".join(reasons) if reasons else ""))
                with st.expander("So sánh chi tiết các sản phẩm gợi ý"):
                    columns_to_show = ["productDisplayName", "articleType", "gender", "subCategory", "baseColour", "season", "usage", "year", "score"]
                    table = candidates[columns_to_show].rename(columns={
                        "productDisplayName": "Tên sản phẩm", "articleType": "Loại", "gender": "Đối tượng",
                        "subCategory": "Danh mục phụ", "baseColour": "Màu", "season": "Mùa",
                        "usage": "Mục đích", "year": "Năm", "score": "Độ tương tự",
                    })
                    st.dataframe(table, width="stretch", hide_index=True)
                    st.download_button("Tải kết quả CSV", table.to_csv(index=False).encode("utf-8-sig"),
                                       file_name="san_pham_goi_y.csv", mime="text/csv")
        else:
            st.info("Hãy chọn ít nhất một sản phẩm mẫu để xem gợi ý.")

with tab_explore:
    a, b, c, d = st.columns(4)
    a.metric("Sản phẩm đã học", f"{metrics['products']:,}")
    b.metric("Nhóm hàng", metrics["main_categories"])
    c.metric("Danh mục phụ", metrics["sub_categories"])
    d.metric("Loại sản phẩm", metrics["article_types"])
    st.caption(f"Kho dữ liệu gốc có {metrics['dataset_products']:,} sản phẩm có ảnh; mô hình hiện được huấn luyện trên tập mẫu tái lập được.")
    st.subheader("Phân bố sản phẩm theo nhóm hàng")
    st.bar_chart(products["masterCategory"].value_counts().head(12))
    st.subheader("Bảng thuộc tính sản phẩm")
    shown = ["productDisplayName", "gender", "masterCategory", "subCategory", "articleType", "baseColour", "season", "year", "usage"]
    st.dataframe(products[shown].rename(columns={"productDisplayName":"Tên sản phẩm","gender":"Đối tượng","masterCategory":"Nhóm hàng","subCategory":"Danh mục phụ","articleType":"Loại","baseColour":"Màu","season":"Mùa","year":"Năm","usage":"Mục đích"}), width="stretch", hide_index=True)

with tab_model:
    st.subheader("Mô hình được huấn luyện như thế nào?")
    st.markdown("1. Chuyển tên sản phẩm thành đặc trưng **TF-IDF**.\n2. Mã hóa **giới tính, nhóm hàng, danh mục phụ, màu, mùa và mục đích sử dụng**.\n3. Tính màu trung bình và phân bố RGB từ **ảnh sản phẩm thật**.\n4. Chuẩn hóa, ghép các nhóm đặc trưng và tìm hàng xóm gần nhất bằng **cosine similarity**.")
    st.write(f"Tách dữ liệu: {metrics['train_products']:,} sản phẩm huấn luyện và {metrics['test_products']:,} sản phẩm kiểm tra.")
    comparison = pd.DataFrame(metrics["comparison"]).T
    comparison = comparison.rename(columns={"precision_at_10":"Precision@10", "hit_rate_at_10":"HitRate@10", "queries":"Số truy vấn"})
    st.dataframe(comparison.style.format({"Precision@10":"{:.1%}", "HitRate@10":"{:.1%}", "Số truy vấn":"{:.0f}"}), width="stretch")
    st.bar_chart(comparison[["Precision@10", "HitRate@10"]])
    st.caption("Precision@10: trung bình tỷ lệ gợi ý cùng loại trong 10 sản phẩm đầu. HitRate@10: tỷ lệ truy vấn có ít nhất một sản phẩm cùng loại.")
    st.caption("Loại sản phẩm (articleType) không nằm trong đầu vào; sản phẩm dùng để truy vấn không nằm trong tập huấn luyện khi đánh giá.")
    st.write(f"Tổng số đặc trưng học máy: {metrics['feature_count']:,}")
    st.info("Dataset không có lịch sử mua hàng, nên chỉ số trên đo độ phù hợp về loại sản phẩm, không phải xác suất người dùng sẽ mua.")
