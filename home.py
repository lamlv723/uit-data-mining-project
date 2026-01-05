import streamlit as st
from sidebar import render_sidebar

# 1. Cấu hình trang (Page Config)
st.set_page_config(
    page_title="Trang chủ - UIT Data Mining", 
    page_icon="⛏️", 
    layout="wide"
)

# 2. Gọi Sidebar
render_sidebar()

# 3. CSS Tùy chỉnh cho trang chủ
st.markdown("""
<style>
    /* Tiêu đề chính */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        color: #ff4b4b;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    
    /* Card giới thiệu từng thuật toán */
    .algo-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        height: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .algo-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-color: #ff4b4b;
    }
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #31333f;
        margin-bottom: 0.5rem;
    }
    .card-desc {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 1rem;
    }
    
    /* Info Box */
    .info-box {
        background-color: #e8f0fe;
        border-left: 5px solid #1a73e8;
        padding: 1rem;
        border-radius: 0.3rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# 4. Nội dung chính
st.markdown('<div class="main-title">⛏️ UIT Data Mining Project</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ứng dụng web demo các thuật toán Khai phá dữ liệu (Data Mining)</div>', unsafe_allow_html=True)

# Phần giới thiệu chung
st.markdown("""
<div class="info-box">
    <b>👋 Chào mừng bạn!</b><br>
    Đây là đồ án môn học Khai phá dữ liệu, được xây dựng bằng <b>Python</b> và <b>Streamlit</b>. 
    Ứng dụng cung cấp giao diện trực quan để chạy và kiểm thử các thuật toán phổ biến trên các tập dữ liệu mẫu hoặc file CSV của bạn.
</div>
""", unsafe_allow_html=True)

st.subheader("🚀 Khám phá các thuật toán")

# Dòng 1: Association Rules & Preprocessing
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="algo-card">
        <div class="card-icon">🛒</div>
        <div class="card-title">Luật Kết Hợp</div>
        <div class="card-desc">
            Thuật toán <b>Apriori</b> giúp tìm ra các tập phổ biến và sinh luật kết hợp từ cơ sở dữ liệu giao dịch.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Thử Apriori", use_container_width=True):
        st.switch_page("pages/1_Apriori.py")

with col2:
    st.markdown("""
    <div class="algo-card">
        <div class="card-icon">🔍</div>
        <div class="card-title">Tập Thô (Reduct)</div>
        <div class="card-desc">
            Sử dụng lý thuyết <b>Rough Sets</b> để tìm tập rút gọn (Reduct) và tập lõi (Core) của dữ liệu.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Thử Reduct", use_container_width=True):
        st.switch_page("pages/6_Reduct.py")

with col3:
    st.markdown("""
    <div class="algo-card">
        <div class="card-icon">🎯</div>
        <div class="card-title">Gom Cụm (K-Means)</div>
        <div class="card-desc">
            Phân nhóm dữ liệu với thuật toán <b>K-Means</b>, hỗ trợ trực quan hóa quá trình di chuyển trọng tâm.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Thử K-Means", use_container_width=True):
        st.switch_page("pages/4_KMeans.py")

st.write("") # Spacer

# Dòng 2: Classification
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div class="algo-card">
        <div class="card-icon">🌳</div>
        <div class="card-title">Cây Quyết Định (ID3)</div>
        <div class="card-desc">
            Xây dựng cây quyết định dựa trên độ lợi thông tin (Information Gain). Hỗ trợ vẽ cây và sinh luật.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Thử ID3", use_container_width=True):
        st.switch_page("pages/2_Decision_Tree.py")

with col5:
    st.markdown("""
    <div class="algo-card">
        <div class="card-icon">🌲</div>
        <div class="card-title">Cây Quyết Định (CART)</div>
        <div class="card-desc">
            Thuật toán cây quyết định sử dụng chỉ số <b>Gini Index</b> để phân lớp dữ liệu.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Thử CART", use_container_width=True):
        st.switch_page("pages/7_CART.py")

with col6:
    st.markdown("""
    <div class="algo-card">
        <div class="card-icon">🧠</div>
        <div class="card-title">Naive Bayes</div>
        <div class="card-desc">
            Mô hình phân lớp dựa trên xác suất thống kê và định lý Bayes. Hỗ trợ kỹ thuật <b>Laplace Smoothing</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Thử Naive Bayes", use_container_width=True):
        st.switch_page("pages/3_Naive_Bayes.py")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #888;">
    Đồ án môn học Khai phá dữ liệu - UIT <br>
    © 2024 - Developed with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)