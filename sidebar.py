import streamlit as st

def render_sidebar():
    # 1. CSS Tùy chỉnh cho Light Theme & Sidebar giống Design
    st.markdown("""
    <style>
        /* Ẩn Sidebar điều hướng mặc định của Streamlit */
        [data-testid="stSidebarNav"] {display: none;}

        /* Tùy chỉnh Sidebar Background (Màu sáng) */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
            border-right: 1px solid #dee2e6;
        }

        /* Tiêu đề chính trong Sidebar */
        .sidebar-title {
            font-size: 2rem;
            font-weight: 700;
            color: #31333f;
            margin-bottom: 1.5rem;
            padding-left: 0.5rem;
        }

        /* Tiêu đề từng Section (như Association Rules, Classification...) */
        .sidebar-section {
            font-size: 0.85rem;
            font-weight: 600;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            padding-left: 0.5rem;
        }
        
        /* Tùy chỉnh các nút Link (st.page_link) */
        div[data-testid="stPageLink-NavLink"] {
            border-radius: 0.375rem;
            padding: 0.5rem 0.75rem;
            color: #31333f;
            transition: background-color 0.2s;
            border: none;
        }
        
        /* Hiệu ứng Hover (Màu xám nhạt) */
        div[data-testid="stPageLink-NavLink"]:hover {
            background-color: #e9ecef;
        }

        /* Trạng thái Active (Đang chọn) - Giữ màu đỏ thương hiệu #ff4b4b */
        div[data-testid="stPageLink-NavLink"][aria-current="page"] {
            background-color: #ff4b4b;
            color: white;
            font-weight: 500;
        }
        
        /* Style cho các icon trong link */
        div[data-testid="stPageLink-NavLink"] svg {
            color: inherit; /* Icon đổi màu theo text */
        }
    </style>
    """, unsafe_allow_html=True)

    # 2. Vẽ nội dung Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-title">📊 Thuật Toán</div>', unsafe_allow_html=True)
        
        # --- Section 1: Association Rules ---
        st.markdown('<div class="sidebar-section">Association Rules</div>', unsafe_allow_html=True)
        st.page_link("pages/1_Apriori.py", label="Tập Phổ Biến & Luật Kết Hợp", icon="🛒")

        # --- Section 2: Classification ---
        st.markdown('<div class="sidebar-section">Classification</div>', unsafe_allow_html=True)
        # Lưu ý: Các file này phải TỒN TẠI trong thư mục pages/ mới chạy được.
        # Nếu chưa tạo file, bạn hãy tạm thời comment lại để không bị lỗi.
        st.page_link("pages/2_Decision_Tree.py", label="Cây Quyết Định (ID3)", icon="🌳")
        st.page_link("pages/3_Naive_Bayes.py", label="Naive Bayes", icon="🧠")

        # --- Section 3: Clustering ---
        st.markdown('<div class="sidebar-section">Clustering</div>', unsafe_allow_html=True)
        st.page_link("pages/4_KMeans.py", label="K-Means / K-Medoids", icon="🎯")
        st.page_link("pages/5_Kohonen.py", label="Mạng Kohonen", icon="🕸️")

        # --- Section 4: Preprocessing ---
        st.markdown('<div class="sidebar-section">Preprocessing</div>', unsafe_allow_html=True)
        st.page_link("pages/6_Reduct.py", label="Tập Thô (Reduct)", icon="🔍")
        
        st.markdown("---")
        st.page_link("home.py", label="Trang chủ", icon="🏠")