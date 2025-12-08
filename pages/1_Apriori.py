# pages/1_📊_Apriori.py
import streamlit as st
import pandas as pd
from algorithms.apriori import Apriori

# --- Cấu hình trang ---
st.set_page_config(page_title="Apriori Algorithm", layout="wide")

# --- CSS Tùy chỉnh (Mô phỏng lại giao diện design.html một chút) ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #fafafa; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1.1rem; color: #a3a8b4;}
    .highlight-box {background-color: rgba(255, 75, 75, 0.1); border-left: 4px solid #ff4b4b; padding: 1rem; border-radius: 0.375rem; margin-bottom: 1.5rem;}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">Tập Phổ Biến & Luật Kết Hợp</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Khai phá các mẫu thường xuyên xuất hiện trong giao dịch</div>', unsafe_allow_html=True)
st.divider()

# --- Info Box ---
st.markdown("""
<div class="highlight-box">
    <b>💡 Giới thiệu thuật toán:</b><br>
    Thuật toán Apriori giúp tìm ra các tập mặt hàng thường được mua cùng nhau. 
    [cite_start]Ví dụ: 80% khách hàng mua bia thì sẽ mua thuốc lá[cite: 1609].
</div>
""", unsafe_allow_html=True)

# --- Layout chia 2 cột: Sidebar control và Main Content ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Cấu hình tham số")
    
    # Upload file
    uploaded_file = st.file_uploader("Nguồn dữ liệu (CSV)", type=['csv'])
    
    # Nếu chưa có file thì dùng file mẫu
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Đang sử dụng dữ liệu mẫu từ Slide")
        df = pd.read_csv("data/apriori_transaction.csv")

    # Hiển thị bảng dữ liệu thô
    with st.expander("👀 Xem dữ liệu đầu vào", expanded=True):
        st.dataframe(df, hide_index=True)

    # Input tham số
    min_supp = st.slider("Min Support (%)", 0, 100, 40) / 100.0
    min_conf = st.slider("Min Confidence (%)", 0, 100, 60) / 100.0

    run_btn = st.button("▶️ Chạy thuật toán", type="primary")

with col2:
    if run_btn:
        with st.spinner('Đang tính toán...'):
            # Gọi thuật toán từ file backend
            model = Apriori(min_support=min_supp, min_confidence=min_conf)
            model.fit(df)
            rules_df = model.generate_rules()

        # Hiển thị kết quả bằng Tabs
        tab1, tab2 = st.tabs(["📦 Tập phổ biến", "🔗 Luật kết hợp"])
        
        with tab1:
            if not model.itemsets:
                st.warning("Không tìm thấy tập phổ biến nào!")
            else:
                # Chuyển đổi itemsets thành DataFrame để hiển thị đẹp
                itemsets_data = []
                for items, supp in model.itemsets.items():
                    itemsets_data.append({
                        "Tập mặt hàng": ', '.join(items),
                        "Kích thước": len(items),
                        "Support": round(supp, 4)
                    })
                st.dataframe(pd.DataFrame(itemsets_data).sort_values(by="Support", ascending=False), use_container_width=True)

        with tab2:
            if rules_df.empty:
                st.warning("Không tìm thấy luật kết hợp nào!")
            else:
                st.dataframe(rules_df, use_container_width=True)
                st.metric("Số lượng luật tìm thấy", len(rules_df))
    else:
        st.info("👈 Hãy nhấn nút 'Chạy thuật toán' ở cột bên trái")