# Hệ thống gợi ý sản phẩm thời trang bằng học máy

Dự án dùng [Fashion Product Images (Small) trên Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small). Bộ dữ liệu có hơn 44.000 sản phẩm với ảnh thật và các trường: giới tính, nhóm hàng, danh mục phụ, loại hàng, màu, mùa, năm, mục đích sử dụng và tên sản phẩm.

## Mô hình

`train_model.py` kết hợp bốn nhóm đặc trưng: TF-IDF từ tên, One-Hot từ thuộc tính phân loại, năm được chuẩn hóa và thống kê màu RGB từ ảnh. Ma trận đặc trưng được chuẩn hóa; Nearest Neighbors dùng cosine similarity để gợi ý. `articleType` không nằm trong đặc trưng và được dùng làm nhãn đánh giá.

Dữ liệu được chia 80% huấn luyện và 20% kiểm tra theo nhóm hàng. Bộ mã hóa chỉ học trên tập huấn luyện. Ba mô hình được so sánh: chỉ tên, chỉ thuộc tính và kết hợp. Kết quả Precision@10 và HitRate@10 lưu trong `models/model_comparison.csv`. Sau đánh giá, mô hình kết hợp được học lại trên toàn bộ sản phẩm để phục vụ giao diện.

## Chạy

```powershell
pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

Mặc định mô hình lấy mẫu 15.000 sản phẩm có ảnh để huấn luyện nhanh trên máy cá nhân. Để huấn luyện toàn bộ sản phẩm, chạy `python train_model.py --full` (cần nhiều thời gian và bộ nhớ hơn).

CSV và ảnh nằm trong `data/fashion/`; tệp huấn luyện nằm trong `models/`. Giao diện cho phép tìm bằng tiếng Việt hoặc tiếng Anh (ví dụ `váy đỏ`, `áo sơ mi`, `giày`), chọn tối đa ba sản phẩm mẫu, lọc theo nhóm hàng, danh mục phụ, loại sản phẩm, đối tượng, màu, mùa, mục đích sử dụng và khoảng năm. Kết quả có ảnh thật, thông tin sản phẩm, lý do trùng thuộc tính, bảng so sánh và nút xuất CSV.

`catalog_search.py` chứa từ điển từ khóa tiếng Việt. Có thể bổ sung từ đồng nghĩa vào `VIETNAMESE_TERMS` khi cần tìm thêm các loại hàng mới.

Chỉ số đánh giá kiểm tra độ phù hợp của sản phẩm gợi ý. Dataset không có lịch sử giao dịch của người dùng để tính khả năng mua hàng thực tế.
