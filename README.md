# UIT Data Mining Project ⛏️

Ứng dụng web demo các thuật toán Khai phá dữ liệu (Data Mining), được xây dựng bằng **Python** và **Streamlit**.

## 🚀 Danh sách Thuật toán

1. **Apriori:** Khai phá luật kết hợp.
2. **ID3 & CART:** Cây quyết định (Decision Tree).
3. **Naive Bayes:** Phân lớp dựa trên xác suất.
4. **K-Means:** Gom cụm dữ liệu.
5. **Reduct (Rough Sets):** Tìm tập rút gọn.

---

## 🛠️ Hướng dẫn Cài đặt & Chạy

Yêu cầu: Máy tính đã cài đặt **Python 3.8+**.

### 🍎 1. Đối với MacOS

**Bước 1:** Mở **Terminal** và di chuyển (cd) vào thư mục chứa code dự án.

**Bước 2:** Tạo môi trường ảo (virtual environment):

```bash
python3 -m venv venv
```

**Bước 3:** Kích hoạt môi trường ảo:

```bash
source venv/bin/activate
```

**Bước 4:** Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

**Bước 5:** Cài đặt Graphviz (để vẽ cây quyết định):

```bash
brew install graphviz
```

*(Nếu chưa có Homebrew, bạn có thể bỏ qua bước này nhưng tính năng vẽ cây có thể bị lỗi)*

**Bước 6:** Chạy ứng dụng:

```bash
streamlit run home.py
```

---

### 🪟 2. Đối với Windows

**Bước 1:** Mở **Command Prompt (cmd)** hoặc **PowerShell** và di chuyển (cd) vào thư mục chứa code dự án.

**Bước 2:** Tạo môi trường ảo:

```bash
python -m venv venv
```

**Bước 3:** Kích hoạt môi trường ảo:

```bash
.\venv\Scripts\activate
```

**Bước 4:** Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

**Bước 5:** Cài đặt Graphviz (để vẽ cây quyết định):

* Tải bộ cài đặt tại: [https://graphviz.org/download/](https://graphviz.org/download/)
* Khi cài đặt, nhớ tích chọn **"Add Graphviz to the system PATH for all users"**.

**Bước 6:** Chạy ứng dụng:

```bash
streamlit run home.py
```

---

## 🌐 Truy cập

Sau khi chạy lệnh `streamlit run home.py`, trình duyệt sẽ tự động mở hoặc bạn truy cập tại:
`http://localhost:8501`

## 📂 Cấu trúc Dự án

```text
├── algorithms/      # Source code logic các thuật toán (Backend)
├── pages/           # Giao diện từng thuật toán (Frontend - Streamlit)
├── data/            # Các file dữ liệu mẫu (.csv)
├── home.py          # Trang chủ ứng dụng
├── sidebar.py       # Thanh điều hướng
└── setup_project.py # Script tạo lại dữ liệu mẫu