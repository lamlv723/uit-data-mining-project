import streamlit as st
import pandas as pd
from algorithms.apriori import Apriori
from sidebar import render_sidebar

# Cấu hình & Sidebar
st.set_page_config(page_title="Apriori", layout="wide")
render_sidebar()

# CSS Tùy chỉnh
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #31333f; margin-bottom: 0.5rem;}
    .highlight-box {background-color: #f0f2f6; border-left: 4px solid #ff4b4b; padding: 1rem; border-radius: 0.375rem; margin-bottom: 1.5rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Tập Phổ Biến & Luật Kết Hợp</div>', unsafe_allow_html=True)

# Hướng dẫn định dạng dữ liệu
st.markdown("""
<div class="highlight-box">
    <b>💡 Lưu ý về dữ liệu:</b><br>
    File CSV cần có 2 cột: <b>Mã giao dịch</b> và <b>Mã hàng</b> (dạng Transaction Format).<br>
    Ví dụ:<br>
    <code>01, i1</code><br>
    <code>01, i2</code><br>
    <code>02, i2</code>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("⚙️ Cấu hình")
    
    # Chọn nguồn dữ liệu
    data_source = st.radio("Nguồn dữ liệu:", ("Dữ liệu mẫu (Slide)", "Tải file CSV"))
    
    df = None
    if data_source == "Dữ liệu mẫu (Slide)":
        try:
            df = pd.read_csv("data/apriori_transaction.csv")
            st.success("Đã tải dữ liệu mẫu.")
        except FileNotFoundError:
            st.error("Chưa tìm thấy file data/apriori_transaction.csv")
    else:
        uploaded_file = st.file_uploader("Upload CSV (2 cột)", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)

    # Hiển thị dữ liệu thô
    if df is not None:
        with st.expander("👀 Xem dữ liệu thô", expanded=True):
            st.dataframe(df, hide_index=True, use_container_width=True)

    # Tham số
    min_supp = st.slider("Min Support", 0.0, 1.0, 0.4, 0.05)
    min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.7, 0.05)
    
    run_btn = st.button("▶️ Chạy thuật toán", type="primary", disabled=(df is None))

with col2:
    st.subheader("📊 Kết quả")
    
    if run_btn and df is not None:
        try:
            # Chạy thuật toán
            model = Apriori(min_support=min_supp, min_confidence=min_conf)
            model.fit(df)
            
            # Lấy kết quả
            itemsets = model.get_itemsets()
            rules = model.generate_rules()
            
            tab1, tab2 = st.tabs(["📦 Tập phổ biến", "🔗 Luật kết hợp"])
            
            with tab1:
                if not itemsets.empty:
                    st.dataframe(itemsets, use_container_width=True)
                    st.metric("Số lượng tập phổ biến", len(itemsets))
                else:
                    st.warning(f"Không tìm thấy tập phổ biến với Min Support = {min_supp}")
            
            with tab2:
                if not rules.empty:
                    st.dataframe(rules, use_container_width=True)
                    st.metric("Số lượng luật", len(rules))
                else:
                    st.warning(f"Không tìm thấy luật với Min Confidence = {min_conf}")
                    
        except Exception as e:
            st.error(f"Lỗi: {e}")