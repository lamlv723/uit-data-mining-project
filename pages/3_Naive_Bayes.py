import streamlit as st
import pandas as pd
from algorithms.naive_bayes import NaiveBayes
from sidebar import render_sidebar

# Cấu hình & Sidebar
st.set_page_config(page_title="Naive Bayes", layout="wide")
render_sidebar()

# CSS & Helper
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #31333f; margin-bottom: 0.5rem;}
    .result-card {background-color: #d4edda; color: #155724; padding: 1rem; border-radius: 0.375rem; margin-top: 1rem; border: 1px solid #c3e6cb;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Phân lớp Naive Bayes</div>', unsafe_allow_html=True)

# Hàm Reset State
def reset_state():
    if 'nb_model' in st.session_state:
        del st.session_state['nb_model']

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("1. Cấu hình Dữ liệu")
    
    # Chọn nguồn dữ liệu
    data_source = st.radio(
        "Nguồn dữ liệu:", 
        ("Dữ liệu mẫu (Weather)", "Tải file CSV"),
        on_change=reset_state
    )
    
    df = None
    if data_source == "Dữ liệu mẫu (Weather)":
        try:
            df = pd.read_csv("data/bayes_weather.csv")
            st.success("Đã tải dữ liệu Weather")
        except: st.error("Lỗi đọc file data.")
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'], on_change=reset_state)
        if uploaded_file:
            df = pd.read_csv(uploaded_file)

    if df is not None:
        with st.expander("👀 Xem dữ liệu thô", expanded=False):
            st.dataframe(df, hide_index=True)
            
        all_cols = df.columns.tolist()
        
        st.write("---")
        target_col = st.selectbox("🎯 Cột kết quả (Target):", all_cols, index=len(all_cols)-1, on_change=reset_state)
        
        # Tùy chọn Laplace
        use_laplace = st.checkbox("Sử dụng làm trơn Laplace", value=False, on_change=reset_state)
        if use_laplace:
            st.caption("Công thức: P(xi|c) = (count + 1) / (total + số giá trị rời rạc)")
        
        if st.button("▶️ Huấn luyện Mô hình", type="primary"):
            model = NaiveBayes(use_laplace=use_laplace)
            model.fit(df, target_col)
            
            st.session_state.nb_model = model
            st.session_state.nb_features = [c for c in df.columns if c != target_col]
            st.session_state.nb_df = df # Lưu để lấy unique values cho form nhập liệu

with col2:
    if 'nb_model' in st.session_state:
        model = st.session_state.nb_model
        priors, likelihoods = model.get_details()
        
        st.subheader("2. Tham số Mô hình")
        
        tab1, tab2 = st.tabs(["📊 Xác suất Tiên nghiệm P(C)", "📈 Xác suất Có điều kiện P(X|C)"])
        
        with tab1:
            # Hiển thị P(Ci)
            prior_df = pd.DataFrame(list(priors.items()), columns=["Lớp (Class)", "Xác suất P(C)"])
            st.table(prior_df)
            
        with tab2:
            # Hiển thị P(Xk|Ci) cho từng thuộc tính
            feature_selected = st.selectbox("Chọn thuộc tính để xem:", list(likelihoods.keys()))
            
            if feature_selected:
                data_dict = likelihoods[feature_selected]
                # Chuyển đổi dict lồng nhau thành DataFrame
                # Index là Class, Columns là Giá trị thuộc tính
                df_view = pd.DataFrame(data_dict).T 
                st.write(f"**P({feature_selected} | Lớp)**")
                st.dataframe(df_view.style.format("{:.4f}"))

        st.divider()
        
        # --- PHẦN DỰ ĐOÁN ---
        st.subheader(f"3. Dự đoán: {target_col}")
        
        with st.form("nb_predict_form"):
            st.caption("Nhập giá trị cho mẫu mới:")
            user_inputs = {}
            input_cols = st.columns(2)
            
            train_df = st.session_state.nb_df
            features = st.session_state.nb_features
            
            for i, col_name in enumerate(features):
                unique_vals = train_df[col_name].unique()
                with input_cols[i % 2]:
                    user_inputs[col_name] = st.selectbox(f"{col_name}", unique_vals)
            
            predict_btn = st.form_submit_button("🔮 Dự đoán ngay")
            
            if predict_btn:
                result, posteriors = model.predict(user_inputs)
                
                st.markdown(f"""
                <div class="result-card">
                    <b>Kết quả dự đoán: {result}</b>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("🔎 Xem chi tiết tính toán"):
                    for c, info in posteriors.items():
                        st.markdown(f"**Lớp {c}:**")
                        st.code(f"{info['details']} \n= {info['score']:.6f}")
    
    elif df is None:
        st.info("👈 Hãy chọn dữ liệu ở cột bên trái.")
    else:
        st.info("👈 Nhấn nút 'Huấn luyện' để xem các bảng xác suất.")