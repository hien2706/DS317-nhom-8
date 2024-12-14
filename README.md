
---

# DS317 - Nhóm 8

## Tên Đề Tài
**DỰ ĐOÁN KẾT QUẢ HỌC TẬP CỦA CÁC MÔN HỌC MÀ SINH VIÊN LỰA CHỌN Ở HỌC KỲ TIẾP THEO**

---

## Thời Gian Thực Hiện
Trong thời gian học phần **Khai Phá Dữ Liệu Trong Doanh Nghiệp**, học kỳ 1, năm học 2024-2025.

---

## Nhóm Thực Hiện
Nhóm 8 lớp DS317.P11 gồm 7 thành viên:

Họ và Tên           | MSSV  
--------------------------|------------
Lê Tuấn Đạt             | 21520699  
Ngô Gia Lâm             | 21521054  
Phạm Lê Thành Phát      | 21521262  
Phạm Huỳnh Thiên Phú    | 21521278  
Nguyễn Hồng Cát Thy     | 21522665  
Trần Đại Hiển           | 22520426  
Trần Lương Vân Nhi      | 22521044  

---

## Mô Tả Dự Án
Dự án tập trung vào việc xây dựng mô hình dự đoán kết quả học tập của các môn học mà sinh viên dự kiến lựa chọn ở học kỳ tiếp theo. 

Dữ liệu và phân tích sẽ được thực hiện trong khuôn khổ môn học, ứng dụng các kỹ thuật khai phá dữ liệu và dự đoán nhằm cung cấp các kết quả chính xác và có ý nghĩa thực tiễn.

---

## Cấu Trúc Dự Án

```
.
├── 1.merge_into_raw_data.ipynb
├── 2.feature-engineering_hocky_namhoc_counter_fe.ipynb
├── 3.scaling_encoding_splitting_dataset.ipynb
├── 4.split_train_val_test.ipynb
├── 5.modelling.ipynb
├── data
│   ├── processed
│   │   └── processed_data.zip
│   └── raw
│       └── raw_data.zip
├── LICENSE
├── notebooks
│   ├── Chuẩn bị dữ liệu
│   │   ├── 1.chuan-bi-merge-raw-data.ipynb
│   │   ├── 2.chuan-bi-sv_dtbhk_stchk.ipynb
│   │   ├── 3.merge-1-and-2-to-raw-merged-data.ipynb
│   │   └── 4.feature-engineering_hocky_namhoc_counter_fe.ipynb
│   ├── Huấn luyện mô hình
│   │   ├── 1.scaling_encoding_splitting_dataset.ipynb
│   │   ├── 2.split_train_val_test.ipynb
│   │   └── 3.modelling.ipynb
│   └── Khám phá dữ liệu
│       ├── Khám phá các thuộc tính của dataset.ipynb
│       ├── Khám phá và đánh giá dữ liệu thô.ipynb
│       ├── Kiểm chứng độ hiệu quả khi feature engineering thêm cột mới.ipynb
│       └── Trực quan hóa dữ liệu.ipynb
└── README.md
```

---

## Hướng Dẫn Sử Dụng

1. **Dữ Liệu:**
   - Thư mục `data/raw` chứa dữ liệu thô (`raw_data.zip`).
   - Thư mục `data/processed` chứa dữ liệu đã qua xử lý (`processed_data.zip`).

2. **Các Notebook Chính:**
   - Chuẩn bị dữ liệu: `notebooks/Chuẩn bị dữ liệu/`
   - Huấn luyện mô hình: `notebooks/Huấn luyện mô hình/`
   - Khám phá dữ liệu: `notebooks/Khám phá dữ liệu/`

3. **Chạy Toàn Bộ Pipeline:**
   - Thực hiện theo thứ tự các file notebook chính:
     1. `1.merge_into_raw_data.ipynb`
     2. `2.feature-engineering_hocky_namhoc_counter_fe.ipynb`
     3. `3.scaling_encoding_splitting_dataset.ipynb`
     4. `4.split_train_val_test.ipynb`
     5. `5.modelling.ipynb`

---

