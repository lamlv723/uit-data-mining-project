import streamlit as st
import pandas as pd
from algorithms.reduct import RoughSets
from sidebar import render_sidebar

# Cấu hình
st.set_page_config(page_title="Tập thô (Rough Sets)", layout="wide")
render_sidebar()

# CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #31333f; margin-bottom: 0.5rem;}
    .reduct-tag {display: inline-block; background-color: #e8f0fe; color: #1a73e8; padding: 0.2rem 0.6rem; border-radius: 1rem; margin-right: 0.5rem; margin-bottom: 0.5rem; border: 1px solid #d2e3fc;}
    .core-tag {display: inline-block; background-color: #fce8e6; color: #c5221f; padding: 0.2rem 0.6rem; border-radius: 1rem; margin-right: 0.5rem; border: 1px solid #fad2cf; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Tập Thô (Reduct)</div>', unsafe_allow_html=True)

def reset_state():
    if 'reduct_model' in st.session_state:
        del st.session_state['reduct_model']

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("1. Cấu hình Dữ liệu")
    
    data_source = st.radio(
        "Nguồn dữ liệu:", 
        ("Ví dụ 1 (Sunburned)", "Ví dụ 2 (Tuyển dụng)", "Tải file CSV"),
        on_change=reset_state
    )
    
    df = None
    try:
        if data_source == "Ví dụ 1 (Sunburned)":
            df = pd.read_csv("data/reduct_attributes.csv")
            st.success("Đã tải dữ liệu Sunburned")
        elif data_source == "Ví dụ 2 (Tuyển dụng)":
            df = pd.read_csv("data/reduct_recruitment.csv")
            st.success("Đã tải dữ liệu Tuyển dụng")
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'], on_change=reset_state)
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
    except FileNotFoundError:
        st.error("Không tìm thấy file dữ liệu. Hãy kiểm tra thư mục data/.")

    if df is not None:
        with st.expander("👀 Xem dữ liệu thô", expanded=False):
            st.dataframe(df, hide_index=True)
            
        all_cols = df.columns.tolist()
        
        st.write("---")
        target_col = st.selectbox("🎯 Thuộc tính Quyết định (Decision):", all_cols, index=len(all_cols)-1, on_change=reset_state)
        id_col = st.selectbox("🚫 Cột ID (Bỏ qua):", ["(None)"] + all_cols, index=1, on_change=reset_state)
        
        if st.button("▶️ Tìm Reduct & Sinh Luật", type="primary"):
            ignore_col = None if id_col == "(None)" else id_col
            
            model = RoughSets()
            model.fit(df, target_col, ignore_col)
            
            st.session_state.reduct_model = model
            st.session_state.reduct_df = df
            st.session_state.reduct_ignore_col = ignore_col
            st.session_state.reduct_target = target_col

with col2:
    if 'reduct_model' in st.session_state:
        model = st.session_state.reduct_model
        data_df = st.session_state.reduct_df
        ignore_col = st.session_state.reduct_ignore_col
        target_col = st.session_state.reduct_target
        
        st.subheader("2. Kết quả Phân tích")
        
        # Hiển thị độ phụ thuộc
        st.info(f"📊 Độ phụ thuộc (Dependency): **{model.dependency:.4f}**")
        
        # TAB HIỂN THỊ
        tab1, tab2 = st.tabs(["✂️ Tập Rút Gọn (Reducts) & Core", "📜 Các Luật (Rules)"])
        
        with tab1:
            # Reducts
            st.write("### Các tập rút gọn (Reducts)")
            if model.reducts:
                st.write(f"Tìm thấy **{len(model.reducts)}** tập rút gọn:")
                for i, reduct in enumerate(model.reducts):
                    reduct_str = "".join([f"<span class='reduct-tag'>{attr}</span>" for attr in reduct])
                    st.markdown(f"{i+1}. {reduct_str}", unsafe_allow_html=True)
            else:
                st.warning("Không tìm thấy Reduct.")
                
            st.divider()
            
            # Core
            st.write("### Tập lõi (Core)")
            if model.core:
                core_str = "".join([f"<span class='core-tag'>{attr}</span>" for attr in model.core])
                st.markdown(f"**Attributes:** {core_str}", unsafe_allow_html=True)
            else:
                st.info("Tập lõi rỗng.")
        
        with tab2:
            st.write("### Danh sách Luật sinh từ Reducts")
            rules_df = model.get_rules(data_df, target_col, ignore_col)
            
            if not rules_df.empty:
                st.dataframe(rules_df, use_container_width=True, hide_index=True)
                st.caption(f"Tổng cộng: {len(rules_df)} luật được sinh ra.")
            else:
                st.info("Không sinh được luật nào.")

    elif df is None:
        st.info("👈 Hãy chọn dữ liệu ở cột bên trái.")
    else:
        st.info("👈 Nhấn nút để chạy thuật toán.")