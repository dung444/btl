# Hệ thống gợi ý và phân loại sản phẩm thời trang bằng học máy (Machine Learning)

**Sinh viên thực hiện:** Trần Đức Lương  
**Đề tài:** Ứng dụng mô hình học máy về hệ thống gợi ý sản phẩm thời trang.  
**Bộ dữ liệu sử dụng:** [Fashion Product Images (Small) trên Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small).

---

## 📁 Cấu trúc dự án

Dự án được tổ chức gọn gàng, chuẩn theo cấu trúc đồ án Machine Learning:

```text
├── .gitignore                      # Cấu hình bỏ qua môi trường ảo, file tạm và kho ảnh nặng
├── README.md                       # Tài liệu thuyết minh và hướng dẫn chạy đồ án
├── requirements.txt                # Danh sách thư viện Python cần thiết
├── .streamlit/
│   └── config.toml                 # Cấu hình theme và giao diện Streamlit
├── app/
│   └── ML_product_recommender.ipynb # Jupyter Notebook thực nghiệm chi tiết từng bước & huấn luyện model
├── data/
│   ├── huongdan.txt                # Hướng dẫn tải và thiết lập tập dữ liệu ảnh từ Kaggle
│   └── fashion/
│       ├── styles.csv              # File dữ liệu thông tin sản phẩm
│       └── images/                 # Kho ảnh sản phẩm (.jpg)
├── demo/
│   ├── app.py                      # Ứng dụng Web Demo Streamlit
│   └── catalog_search.py           # Module tìm kiếm thông minh từ khóa tiếng Việt
└── models/
    ├── fashion_model.joblib        # Mô hình KNN & ma trận đặc trưng được lưu sau khi train
    ├── products.pkl                # Dữ liệu sản phẩm đã tiền xử lý
    ├── metrics.json                # Kết quả đánh giá mô hình
    └── model_comparison.csv        # Bảng so sánh độ chính xác của các mô hình
```

---

## 🧠 Các nội dung thực hiện trong Notebook (`app/ML_product_recommender.ipynb`)

Notebook được thiết kế theo đúng quy chuẩn các bài tập thực hành học phần Machine Learning:

1. **Khám phá dữ liệu:** Đọc file dữ liệu `styles.csv`, kiểm tra kiểu dữ liệu, thông tin thiếu (`isnull().sum()`), thống kê mô tả.
2. **Làm sạch dữ liệu:** Loại bỏ bản ghi thiếu id, trùng lặp và liên kết với file ảnh thật trong thư mục `images/`.
3. **Trực quan hóa (EDA):** Biểu đồ phân bố các loại sản phẩm (Article Type) và đối tượng (Gender) bằng Matplotlib.
4. **Tiền xử lý đặc trưng:**
   - Mã hóa các thuộc tính chữ (`gender`, `masterCategory`, `subCategory`, `baseColour`, `season`, `usage`) bằng `LabelEncoder`.
   - Chuẩn hóa thuộc tính số bằng `StandardScaler`.
   - Chia tập dữ liệu `train_test_split` (80% Train, 20% Test) phân tầng theo nhãn.
5. **Huấn luyện và so sánh các mô hình học máy:**
   - **K-Nearest Neighbors (KNN)**
   - **Decision Tree**
   - **Random Forest**
   - **Naive Bayes (GaussianNB)**
6. **Đánh giá hiệu quả:** Đo lường qua `accuracy_score`, `classification_report`, lập bảng so sánh và vẽ biểu đồ cột.
7. **Ứng dụng gợi ý sản phẩm:** Sử dụng giải thuật hàng xóm gần nhất của KNN (`kneighbors`) để tìm ra 5 sản phẩm tương tự nhất với sản phẩm mẫu.
8. **Xuất mô hình:** Lưu model và dữ liệu vào thư mục `models/` bằng `joblib` để Web Demo sử dụng.

---

## 🚀 Hướng dẫn chạy dự án

### 1. Cài đặt các thư viện cần thiết
Mở terminal trong thư mục dự án và chạy lệnh:
```powershell
pip install -r requirements.txt
```

### 2. Tải và cài đặt dữ liệu ảnh
Xem hướng dẫn chi tiết tại [data/huongdan.txt](data/huongdan.txt) để tải bộ dữ liệu ảnh từ Kaggle và giải nén vào thư mục `data/fashion/images/`.

### 3. Huấn luyện mô hình (Training)
Mở file [app/ML_product_recommender.ipynb](app/ML_product_recommender.ipynb) trong VS Code hoặc Jupyter Notebook:
- Chọn kernel Python 3.
- Bấm **Run All** để chạy tuần tự tất cả các ô từ đầu đến cuối.
- Khi chạy xong ô cuối cùng, các file model sẽ tự động được lưu vào thư mục `models/`.

### 4. Khởi chạy Giao diện Web Demo
Chạy lệnh Streamlit từ terminal:
```powershell
streamlit run demo/app.py
```
Giao diện web sẽ tự động mở trên trình duyệt, cho phép bạn tìm kiếm sản phẩm theo từ khóa tiếng Việt, chọn sản phẩm mẫu và nhận các gợi ý thời trang tương tự kèm ảnh trực quan.
