# UIT Data Mining Project ⛏️

Ứng dụng web demo các thuật toán Khai phá dữ liệu (Data Mining), được xây dựng bằng **Python** và **Streamlit**. Dự án bao gồm các thuật toán phân lớp, gom cụm và khai phá luật kết hợp phổ biến.

## 🚀 Danh sách Thuật toán

1.  **Apriori:** Khai phá luật kết hợp (Association Rules).
2.  **ID3 Decision Tree:** Cây quyết định (có vẽ biểu đồ cây & sinh luật).
3.  **Naive Bayes:** Phân lớp dựa trên xác suất (hỗ trợ làm trơn Laplace).
4.  **K-Means:** Gom cụm dữ liệu (trực quan hóa từng bước di chuyển tâm cụm).
5.  **Reduct (Rough Sets):** Tìm tập rút gọn và tập lõi (Core) của dữ liệu.

## 🛠️ Cài đặt & Chạy

Yêu cầu: Python 3.8+

1.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Lưu ý: Cần cài đặt Graphviz trên máy để vẽ cây quyết định)*

2.  **Chạy ứng dụng:**
    ```bash
    streamlit run home.py
    ```

3.  **Truy cập:** Mở trình duyệt tại địa chỉ `http://localhost:8501`.

## 📂 Cấu trúc Dự án

```text
├── algorithms/      # Source code logic các thuật toán (Backend)
├── pages/           # Giao diện từng thuật toán (Frontend - Streamlit)
├── data/            # Các file dữ liệu mẫu (.csv)
├── home.py          # Trang chủ ứng dụng
├── sidebar.py       # Thanh điều hướng
└── setup_project.py # Script tạo lại dữ liệu mẫu