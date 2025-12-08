import streamlit as st
from sidebar import render_sidebar

st.set_page_config(page_title="Đồ án Data Mining", layout="wide")

# Gọi Sidebar tùy chỉnh
render_sidebar()

st.title("Chào mừng đến với Đồ án Data Mining")
st.write("Chọn một thuật toán từ menu bên trái để bắt đầu.")