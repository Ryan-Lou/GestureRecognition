import cv2
import numpy as np
from typing import Tuple, Optional
import yaml

class CameraManager:
    """摄像头管理类，负责摄像头的初始化和图像获取"""
    
    def __init__(self, config_path: str):
        """
        初始化摄像头管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.camera = None
        self._init_camera()
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _init_camera(self) -> None:
        """初始化摄像头"""
        camera_config = self.config['camera']
        self.camera = cv2.VideoCapture(camera_config['id'])
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['width'])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['height'])
        self.camera.set(cv2.CAP_PROP_FPS, camera_config['fps'])
        
        if not self.camera.isOpened():
            raise RuntimeError("无法打开摄像头")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        读取一帧图像
        
        Returns:
            np.ndarray: 图像帧，如果读取失败则返回None
        """
        if self.camera is None:
            return None
            
        ret, frame = self.camera.read()
        if not ret:
            return None
            
        return frame
    
    def release(self) -> None:
        """释放摄像头资源"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None 