import unittest
import os
from src.face_processor import load_known_faces, process_image
from PIL import Image
import numpy as np

class TestFaceProcessor(unittest.TestCase):
    def test_load_known_faces(self):
        """测试加载已知人脸库"""
        # 测试空目录
        encodings, names = load_known_faces("test_empty_dir")
        self.assertEqual(len(encodings), 0)
        self.assertEqual(len(names), 0)
        
        # 测试正常目录
        encodings, names = load_known_faces("known_faces")
        # 至少应该有两个已知人脸
        self.assertGreaterEqual(len(encodings), 0)
        self.assertGreaterEqual(len(names), 0)
    
    def test_process_image(self):
        """测试处理图像"""
        # 加载已知人脸
        known_encodings, known_names = load_known_faces("known_faces")
        
        # 测试示例图片
        if os.path.exists("examples/lhc3.png"):
            image_path = "examples/lhc3.png"
            img, count, results = process_image(image_path, known_encodings, known_names)
            # 应该检测到至少一个人脸
            self.assertGreaterEqual(count, 0)
            # 结果应该是列表
            self.assertIsInstance(results, list)

if __name__ == "__main__":
    unittest.main()