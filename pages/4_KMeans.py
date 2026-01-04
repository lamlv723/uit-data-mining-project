import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from algorithms.kmeans import KMeansClustering
from sidebar import render_sidebar

# Cấu hình
st.set_page_config(page_title="K-Means Clustering", layout="wide")
render_sidebar()

# CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #31333f; margin-bottom: 0.5rem;}
    .step-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Gom cụm K-Means</div>', unsafe_allow_html=True)

# Hàm reset
def reset_state():
    if 'kmeans_model' in st.session_state:
        del st.session_state['kmeans_model']

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("1. Cấu hình Dữ liệu")
    
    # Chọn nguồn dữ liệu
    data_source = st.radio(
        "Nguồn dữ liệu:", 
        ("Dữ liệu mẫu", "Tải file CSV"),
        on_change=reset_state
    )
    
    df = None
    if data_source == "Dữ liệu mẫu":
        try:
            df = pd.read_csv("data/kmeans_points.csv")
            st.success("Đã tải dữ liệu mẫu.")
        except: st.error("Lỗi đọc file data/kmeans_points.csv")
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'], on_change=reset_state)
        if uploaded_file:
            df = pd.read_csv(uploaded_file)

    if df is not None:
        with st.expander("👀 Xem dữ liệu thô", expanded=True):
            st.dataframe(df, hide_index=True)
            
        st.write("---")
        # Chọn số cụm K
        k_value = st.slider("Chọn số cụm (K):", min_value=1, max_value=5, value=2, on_change=reset_state)
        
        if st.button("▶️ Chạy Gom cụm", type="primary"):
            model = KMeansClustering(k=k_value)
            steps = model.fit(df)
            st.session_state.kmeans_steps = steps
            st.rerun()

with col2:
    if 'kmeans_steps' in st.session_state:
        steps = st.session_state.kmeans_steps
        total_steps = len(steps)
        
        st.subheader("2. Kết quả & Trực quan hóa")
        
        # Thanh trượt chọn bước
        if total_steps > 1:
            step_idx = st.slider("Chọn bước thực thi (Iteration):", 1, total_steps, total_steps) - 1
        else:
            step_idx = 0
            
        current_step = steps[step_idx]
        
        # Lấy dữ liệu bước hiện tại
        centroids = current_step['centroids']
        labels = current_step['labels']
        data_df = current_step['data']
        
        # Thêm cột Cluster vào dataframe để vẽ
        plot_df = data_df.copy()
        plot_df['Cluster'] = labels
        plot_df['Cluster'] = plot_df['Cluster'].astype(str) # Để tô màu rời rạc
        
        # Xác định tên 2 cột tọa độ
        # File mẫu: Point, X, Y -> Lấy X, Y
        numeric_cols = plot_df.select_dtypes(include=['float64', 'int64']).columns
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        
        # --- VẼ BIỂU ĐỒ ---
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 1. Vẽ các điểm dữ liệu
        sns.scatterplot(
            data=plot_df, x=x_col, y=y_col, 
            hue='Cluster', style='Cluster', 
            s=200, palette='viridis', ax=ax, zorder=2
        )
        
        # 2. Vẽ các trọng tâm (Centroids)
        # Centroids là numpy array, cột 0 là x_col, cột 1 là y_col
        ax.scatter(
            centroids[:, 0], centroids[:, 1], 
            c='red', s=400, marker='X', label='Centroids', zorder=3
        )
        
        # Label cho điểm (nếu có cột tên, ví dụ cột đầu tiên)
        first_col = plot_df.columns[0]
        if plot_df[first_col].dtype == 'object':
            for i, txt in enumerate(plot_df[first_col]):
                ax.annotate(txt, (plot_df[x_col][i], plot_df[y_col][i]), xytext=(5, 5), textcoords='offset points')

        ax.set_title(f"Iteration {current_step['iteration']}", fontsize=15)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        st.pyplot(fig)
        
        # --- HIỂN THỊ CHI TIẾT ---
        with st.expander("🔎 Chi tiết tọa độ trọng tâm"):
            st.write(f"**Trọng tâm tại bước {step_idx + 1}:**")
            centroid_df = pd.DataFrame(centroids, columns=[x_col, y_col])
            centroid_df.index.name = "Cluster ID"
            st.dataframe(centroid_df)

    elif df is None:
        st.info("👈 Hãy chọn dữ liệu ở cột bên trái.")
    else:
        st.info("👈 Nhấn nút 'Chạy Gom cụm' để bắt đầu.")