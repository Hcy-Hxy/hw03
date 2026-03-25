import streamlit as st
import numpy as np
from PIL import Image
import os
import pandas as pd

# 导入核心处理模块
from src.face_processor import load_known_faces, process_image

# ================= 1. 页面基本设置 =================
st.set_page_config(
    page_title="人脸识别系统 HW03", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 2. 应用标题和介绍 =================
st.title("🧑‍💻 HW03: 基于人脸识别的图像处理系统")
st.markdown("""
本系统基于 `face_recognition` 库和 `Streamlit` 框架，实现了人脸检测、识别和可视化展示功能。

**核心功能：**
- 人脸检测与定位
- 人脸识别与比对
- 多人脸同时处理
- 识别阈值调节
- 置信度显示
- 详细的识别结果表格
- 完善的错误处理
- 响应式设计

**使用说明：**
1. 在侧边栏调节识别容忍度
2. 选择图片来源（本地上传或系统示例）
3. 上传图片或选择示例图片
4. 查看识别结果
""")
st.markdown("---")

# ================= 3. 侧边栏与系统初始化 =================
with st.sidebar:
    st.header("⚙️ 算法调节")
    # 使用 session state 确保滑动条变化时实时更新
    if 'tolerance' not in st.session_state:
        st.session_state.tolerance = 0.6
    
    tolerance = st.slider(
        "识别容忍度 (Tolerance)", 
        min_value=0.3, 
        max_value=0.8, 
        value=st.session_state.tolerance, 
        step=0.05,
        help="值越低要求越严格，默认0.6。"
    )
    st.session_state.tolerance = tolerance
    st.markdown("---")

    # 加载已知人脸
    st.header("📁 人脸库状态")
    try:
        known_encodings, known_names = load_known_faces()
        if known_names:
            st.success(f"✅ 成功加载 {len(known_names)} 个已知特征。")
            st.write(f"**已知人脸：** {', '.join(known_names)}")
        else:
            st.warning("⚠️ `known_faces` 目录为空，当前仅支持人脸检测。")
    except Exception as e:
        st.error(f"❌ 加载人脸库失败: {e}")
        known_encodings, known_names = [], []
    st.markdown("---")

    # 系统信息
    st.header("ℹ️ 系统信息")
    st.write("**版本：** 1.0.0")
    st.write("**依赖：** face_recognition, streamlit, numpy, Pillow")

# ================= 4. 前端交互与展示 =================
st.header("📸 图片选择")
upload_source = st.radio(
    "选择待检测图像:", 
    ["📂 本地上传", "🖼️ 系统示例"], 
    horizontal=True,
    help="选择图片来源"
)

selected_image = None
if "本地上传" in upload_source:
    selected_image = st.file_uploader(
        "请上传包含人脸的图片", 
        type=["jpg", "jpeg", "png"],
        help="支持 jpg、jpeg、png 格式的图片"
    )
else:
    if os.path.exists("examples"):
        files = [f for f in os.listdir("examples") if f.endswith(('.jpg', '.png', '.jpeg'))]
        if files:
            choice = st.selectbox(
                "选择示例图片:", 
                files,
                help="从示例图片中选择一张进行测试"
            )
            selected_image = os.path.join("examples", choice)
        else:
            st.error("❌ examples 文件夹为空，请添加示例图片。")
    else:
        st.error("❌ 未找到 examples 文件夹，请创建并添加示例图片。")

# 核心渲染区
if selected_image:
    st.markdown("---")
    st.header("🔍 识别结果")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 原始图片")
        try:
            st.image(selected_image, use_container_width=True)
        except Exception as e:
            st.error(f"❌ 显示图片失败: {e}")
            
    with col2:
        st.subheader("🎯 处理结果")
        try:
            with st.spinner("AI 特征提取中..."):
                res_img, count, results = process_image(selected_image, known_encodings, known_names, tolerance)
                
                # 显示识别结果
                st.image(res_img, use_container_width=True)
                
                st.success(f"✅ 处理完成，共检测到 {count} 张人脸。")
                if results:
                    st.subheader("📊 详细信息")
                    st.dataframe(pd.DataFrame(results), use_container_width=True)
        except Exception as e:
            st.error(f"❌ 处理图片失败: {e}")
            import traceback
            st.code(traceback.format_exc())

# ================= 5. 页脚信息 =================
st.markdown("---")
st.markdown("""
### 注意事项
- 首次运行时会加载人脸库，可能需要几秒钟时间
- 识别精度取决于人脸库中图片的质量和数量
- 对于多人脸图片，识别速度会稍微变慢
- 确保图片光线充足，人脸清晰，以获得最佳识别效果
- 对于分辨率过高的图片，处理速度可能会较慢

### 常见问题
- **安装 dlib 失败**：确保已安装 C++ 编译环境，尝试 `pip install dlib==19.22.99`
- **识别结果不准确**：调整识别阈值，提高人脸库中图片的质量
- **应用运行缓慢**：对于多人脸图片，识别速度会较慢，这是正常现象
- **上传图片失败**：确保上传的图片格式正确（jpg、jpeg、png），且文件大小适中
- **人脸框位置不准确**：确保图片光线充足，人脸清晰可见，且人脸在图片中占比适当
""")
st.markdown("© 2026 人脸识别系统 HW03")
