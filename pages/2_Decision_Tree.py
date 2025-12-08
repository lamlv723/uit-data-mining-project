import streamlit as st
import pandas as pd
import graphviz
from algorithms.id3 import ID3DecisionTree
from sidebar import render_sidebar

st.set_page_config(page_title="Cây Quyết Định (ID3)", layout="wide")
render_sidebar()

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #31333f; margin-bottom: 0.5rem;}
    .highlight-box {background-color: #f0f2f6; border-left: 4px solid #ff4b4b; padding: 1rem; border-radius: 0.375rem; margin-bottom: 1.5rem;}
    .result-card {background-color: #d4edda; color: #155724; padding: 1rem; border-radius: 0.375rem; border: 1px solid #c3e6cb; margin-top: 1rem; text-align: center; font-size: 1.5rem; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Cây Quyết Định (ID3)</div>', unsafe_allow_html=True)

def reset_state():
    """Hàm này sẽ xóa model cũ khi người dùng đổi dữ liệu"""
    if 'id3_model' in st.session_state:
        del st.session_state['id3_model']
    if 'feature_cols' in st.session_state:
        del st.session_state['feature_cols']
    # Xóa cache dữ liệu của Streamlit (nếu có dùng @st.cache)
    st.cache_data.clear()


col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("1. Cấu hình Dữ liệu")
    
    data_source = st.radio(
        "Nguồn dữ liệu:", 
        ("Dữ liệu mẫu (Play Golf)", "Dữ liệu mẫu (Tax Evade)", "Tải file CSV"),
        on_change=reset_state
    )

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
        
        # 1. Chọn cột Target
        st.write("---")
        target_col = st.selectbox("🎯 Cột kết quả (Target):", all_cols, index=len(all_cols)-1)
        
        # 2. Chọn nhiều cột cần bỏ qua (Multiselect)
        # Tự động gợi ý các cột có tên giống ID
        default_drop = [c for c in all_cols if "day" in c.lower() or "_raw" in c.lower() or "tid" in c.lower()]
        
        drop_cols = st.multiselect(
            "🚫 Chọn các cột cần bỏ qua (ID, Nhiễu...):", 
            options=all_cols,
            default=default_drop
        )
        st.caption("Gợi ý: Hãy bỏ các cột mã số (Tid, ID) để tránh học vẹt.")

        if 'id3_model' not in st.session_state:
            st.session_state.id3_model = None
            st.session_state.feature_cols = []

        if st.button("▶️ Huấn luyện & Vẽ cây", type="primary"):
            # Huấn luyện
            model = ID3DecisionTree()
            # Truyền danh sách cột cần bỏ
            model.fit(df, target_col, drop_cols)
            
            st.session_state.id3_model = model
            
            # Lưu lại danh sách các cột đặc trưng (để tạo form dự đoán)
            # Feature = Tất cả cột - Target - DropCols
            features = [c for c in df.columns if c != target_col and c not in drop_cols]
            st.session_state.feature_cols = features
            st.session_state.train_df = df 
            st.rerun()

with col2:
    if st.session_state.id3_model is not None:
        st.subheader("2. Kết quả Phân lớp")
        
        model = st.session_state.id3_model
        dot_data = model.get_graphviz_dot()
        rules_df = model.get_rules()
        
        tab1, tab2 = st.tabs(["🌳 Biểu đồ Cây", "📜 Các Luật Quyết định"])
        
        with tab1:
            if dot_data:
                st.graphviz_chart(dot_data)
            else:
                st.warning("Không thể vẽ cây (Dữ liệu quá đơn điệu hoặc lỗi).")
        
        with tab2:
            if not rules_df.empty:
                rules_df.index += 1
                st.table(rules_df)
            else:
                st.info("Không sinh được luật nào.")
            
        st.divider()
        
        # --- PHẦN DỰ ĐOÁN ---
        st.subheader(f"3. Dự đoán: {target_col}")
        
        with st.form("prediction_form"):
            user_inputs = {}
            input_cols = st.columns(2)
            
            train_df = st.session_state.train_df
            feature_cols = st.session_state.feature_cols
            
            for i, col_name in enumerate(feature_cols):
                unique_vals = train_df[col_name].unique()
                with input_cols[i % 2]:
                    user_inputs[col_name] = st.selectbox(f"{col_name}", unique_vals)
            
            predict_btn = st.form_submit_button("🔮 Dự đoán ngay")
            
            if predict_btn:
                result = model.predict(user_inputs)
                st.markdown(f"""
                <div class="result-card">
                    Kết quả dự đoán: {result}
                </div>
                """, unsafe_allow_html=True)
                
    elif df is None:
        st.info("👈 Hãy chọn dữ liệu ở cột bên trái trước.")
    else:
        st.info("👈 Nhấn nút 'Huấn luyện' để bắt đầu.")