import streamlit as st
import pandas as pd
import graphviz
from algorithms.id3 import ID3DecisionTree
from sidebar import render_sidebar

# Cấu hình & Sidebar
st.set_page_config(page_title="Cây Quyết Định (ID3)", layout="wide")
render_sidebar()

# CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #31333f; margin-bottom: 0.5rem;}
    .highlight-box {background-color: #f0f2f6; border-left: 4px solid #ff4b4b; padding: 1rem; border-radius: 0.375rem; margin-bottom: 1.5rem;}
    .result-card {background-color: #d4edda; color: #155724; padding: 1rem; border-radius: 0.375rem; border: 1px solid #c3e6cb; margin-top: 1rem; text-align: center; font-size: 1.5rem; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Cây Quyết Định (ID3)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="highlight-box">
    <b>💡 Hướng dẫn:</b><br>
    1. Chọn dữ liệu huấn luyện để máy học và vẽ cây.<br>
    2. Sau khi có cây, nhập thông tin vào form bên dưới để dự đoán kết quả.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("1. Huấn luyện mô hình")
    
    # Chọn dữ liệu
    data_source = st.radio("Nguồn dữ liệu:", ("Dữ liệu mẫu (Play Golf)", "Dữ liệu mẫu (Tax Evade)", "Tải file CSV"))
    
    df = None
    if data_source == "Dữ liệu mẫu (Play Golf)":
        try:
            df = pd.read_csv("data/decision_tree_play_golf.csv")
            st.success("Đã tải dữ liệu Play Golf.")
        except: st.error("Lỗi file data.")
    elif data_source == "Dữ liệu mẫu (Tax Evade)":
        try:
            df = pd.read_csv("data/decision_tree_tax.csv")
            st.success("Đã tải dữ liệu Tax Evade.")
        except: st.error("Lỗi file data.")
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)

    if df is not None:
        with st.expander("👀 Xem dữ liệu thô", expanded=False):
            st.dataframe(df, hide_index=True)
            
        all_cols = df.columns.tolist()
        
        # Chọn Target
        target_col = st.selectbox("Cột kết quả (Target):", all_cols, index=len(all_cols)-1)
        
        # Chọn ID để bỏ qua
        potential_id = 0 if "Day" in all_cols[0] or "id" in all_cols[0].lower() else None
        id_col = st.selectbox(
            "Cột ID (Bỏ qua):", 
            ["(None)"] + all_cols, 
            index=0 if potential_id is None else potential_id + 1
        )
        
        # Khởi tạo Session State để lưu model
        if 'id3_model' not in st.session_state:
            st.session_state.id3_model = None
            st.session_state.feature_cols = []

        if st.button("▶️ Huấn luyện & Vẽ cây", type="primary"):
            ignore_col = None if id_col == "(None)" else id_col
            
            # Huấn luyện
            model = ID3DecisionTree()
            model.fit(df, target_col, ignore_col)
            
            # Lưu vào session để dùng cho phần dự đoán
            st.session_state.id3_model = model
            st.session_state.feature_cols = [c for c in df.columns if c != target_col and c != ignore_col]
            st.session_state.train_df = df # Lưu df để lấy giá trị cho selectbox
            st.rerun()

with col2:
    if st.session_state.id3_model is not None:
        st.subheader("2. Cây Quyết Định")
        
        # Vẽ cây
        dot_data = st.session_state.id3_model.get_graphviz_dot()
        if dot_data:
            st.graphviz_chart(dot_data)
        else:
            st.warning("Cây rỗng.")
            
        st.divider()
        
        # --- PHẦN DỰ ĐOÁN ---
        st.subheader("3. Dự đoán kết quả mới")
        st.caption("Chọn các thuộc tính để xem kết quả dự đoán:")
        
        with st.form("prediction_form"):
            user_inputs = {}
            # Tạo lưới 2 cột cho đẹp
            input_cols = st.columns(2)
            
            # Tự động tạo selectbox cho từng thuộc tính
            train_df = st.session_state.train_df
            feature_cols = st.session_state.feature_cols
            
            for i, col_name in enumerate(feature_cols):
                unique_vals = train_df[col_name].unique()
                with input_cols[i % 2]:
                    user_inputs[col_name] = st.selectbox(f"{col_name}", unique_vals)
            
            predict_btn = st.form_submit_button("🔮 Dự đoán ngay")
            
            if predict_btn:
                result = st.session_state.id3_model.predict(user_inputs)
                st.markdown(f"""
                <div class="result-card">
                    Kết quả: {result}
                </div>
                """, unsafe_allow_html=True)
                
    elif df is None:
        st.info("👈 Hãy chọn dữ liệu ở cột bên trái trước.")
    else:
        st.info("👈 Hãy nhấn nút 'Huấn luyện & Vẽ cây' để bắt đầu.")