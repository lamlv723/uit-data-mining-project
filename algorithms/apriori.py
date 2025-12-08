# pages/1_📊_Apriori.py
import streamlit as st
import pandas as pd
from algorithms.apriori import Apriori

# Cấu hình trang
st.set_page_config(page_title="Apriori", page_icon="📊", layout="wide")

# Đường dẫn file data cố định cho thuật toán này
DATA_PATH = "data/apriori_transaction.csv"

# --- Header ---
st.title("📊 Thuật toán Apriori: Tập phổ biến & Luật kết hợp")
st.markdown("---")

# --- Layout ---
col1, col2 = st.columns([1, 2], gap="large")

# --- Cột trái: Tham số & Dữ liệu ---
with col1:
    st.subheader("1. Dữ liệu & Tham số")
    
    # Hiển thị dữ liệu thô
    try:
        df = pd.read_csv(DATA_PATH)
        st.caption(f"Đang sử dụng dữ liệu từ: `{DATA_PATH}`")
        st.dataframe(df, hide_index=True, use_container_width=True)
    except FileNotFoundError:
        st.error(f"Không tìm thấy file {DATA_PATH}. Hãy chạy setup_project.py trước!")
        st.stop()

    st.write("---")
    
    # Form nhập tham số
    with st.form("apriori_form"):
        min_supp = st.slider("Min Support (Độ phổ biến tối thiểu)", 0.0, 1.0, 0.4, 0.05)
        min_conf = st.slider("Min Confidence (Độ tin cậy tối thiểu)", 0.0, 1.0, 0.7, 0.05)
        
        submitted = st.form_submit_button("▶️ Chạy thuật toán")

# --- Cột phải: Kết quả ---
with col2:
    st.subheader("2. Kết quả phân tích")
    
    if submitted:
        # Gọi thuật toán
        model = Apriori(min_support=min_supp, min_confidence=min_conf)
        model.fit(DATA_PATH)
        
        # Lấy kết quả
        df_itemsets = model.get_itemsets()
        df_rules = model.get_rules()
        
        # Hiển thị bằng Tab
        tab1, tab2 = st.tabs(["📦 Tập phổ biến (Frequent Itemsets)", "🔗 Luật kết hợp (Rules)"])
        
        with tab1:
            if not df_itemsets.empty:
                st.info(f"Tìm thấy {len(df_itemsets)} tập phổ biến.")
                st.dataframe(df_itemsets, use_container_width=True, height=400)
            else:
                st.warning("Không tìm thấy tập phổ biến nào với ngưỡng Support này.")
                
        with tab2:
            if not df_rules.empty:
                st.info(f"Tìm thấy {len(df_rules)} luật kết hợp.")
                st.dataframe(df_rules, use_container_width=True)
            else:
                st.warning("Không tìm thấy luật nào với ngưỡng Confidence này.")
    else:
        st.info("👈 Nhấn nút 'Chạy thuật toán' để xem kết quả.")