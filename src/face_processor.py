import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import os

def load_known_faces(known_faces_dir="known_faces"):
    """加载已知人脸库"""
    known_encodings = []
    known_names = []
    
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
        return [], []
    
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(known_faces_dir, filename)
            try:
                # 加载图片
                img = face_recognition.load_image_file(filepath)
                # 提取人脸特征
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(os.path.splitext(filename)[0])
            except Exception as e:
                print(f"加载 {filename} 失败: {e}")
    
    return known_encodings, known_names

def process_image(image_file, known_encodings, known_names, tolerance=0.6):
    """处理图像，检测和识别人脸"""
    # 加载图像
    img = Image.open(image_file)
    img = ImageOps.exif_transpose(img)  # 防止手机照片侧翻
    img = img.convert('RGB')
    
    # 转换为 numpy 数组
    img_array = np.array(img)
    
    # 检测人脸位置
    face_locations = face_recognition.face_locations(img_array)
    face_encodings = face_recognition.face_encodings(img_array, face_locations)
    
    # 准备画板
    draw = ImageDraw.Draw(img)
    results_info = []
    
    # 遍历每个人脸
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        confidence = 0.0
        
        # 比对已知人脸
        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_idx = np.argmin(distances)
            if distances[best_match_idx] <= tolerance:
                name = known_names[best_match_idx]
                confidence = (1 - distances[best_match_idx]) * 100
        
        # 绘制人脸框
        box_color = "#00FF00" if name != "Unknown" else "#FF0000"
        line_width = max(2, int(img.width * 0.005))
        draw.rectangle(((left, top), (right, bottom)), outline=box_color, width=line_width)
        
        # 绘制名字标签
        label = f"{name} ({confidence:.1f}%)" if name != "Unknown" else "Unknown"
        text_bg_height = max(20, int(img.height * 0.025))
        draw.rectangle(((left, bottom), (right, bottom + text_bg_height)), fill=box_color)
        draw.text((left + 5, bottom + 2), label, fill="black")
        
        # 记录结果
        results_info.append({
            "人物": name,
            "置信度": f"{confidence:.1f}%" if name != "Unknown" else "-",
            "位置": f"上:{top}, 右:{right}, 下:{bottom}, 左:{left}"
        })
    
    del draw
    return img, len(face_locations), results_info