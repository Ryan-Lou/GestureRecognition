import cv2
import yaml
from typing import Tuple, Optional, Dict, Any

class CameraManager:
    """
    相机管理类
    
    负责相机的初始化、配置和图像捕获。提供统一的接口控制相机设备，
    支持从配置文件加载相机参数，包括分辨率和帧率等设置。
    """
    
    def __init__(self, config_path: str):
        """
        初始化相机管理器
        
        加载相机配置并准备相机设备，但不会立即启动相机。
        相机必须通过调用init_camera方法才能开始使用。
        
        Args:
            config_path: 配置文件路径，YAML格式
        """
        self.config = self._load_config(config_path)
        self.width = self.config['camera']['width']
        self.height = self.config['camera']['height']
        self.fps = self.config['camera']['fps']
        self.cap = None
        self.camera_index = self.config['camera']['camera_index']
        self.flip_horizontal = self.config['camera'].get('flip_horizontal', False)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        从指定路径加载YAML格式的配置文件，用于设置相机参数。
        
        Args:
            config_path: 配置文件的路径
            
        Returns:
            Dict[str, Any]: 包含配置参数的字典
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def init_camera(self) -> bool:
        """
        初始化相机
        
        根据配置打开相机设备，并设置分辨率和帧率等参数。
        如果相机已经初始化，会先关闭现有连接再重新初始化。
        
        Returns:
            bool: 初始化是否成功
        """
        # 如果相机已经初始化，先关闭
        if self.cap is not None:
            self.release()
            
        # 初始化相机
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # 设置相机参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # 检查相机是否成功打开
        if not self.cap.isOpened():
            print("无法打开相机")
            return False
            
        return True
    
    def read_frame(self) -> Tuple[bool, Optional[Any]]:
        """
        读取一帧图像
        
        从相机捕获一帧图像，并根据配置决定是否水平翻转图像。
        在返回图像前会进行基本的错误检查。
        
        Returns:
            Tuple[bool, Optional[Any]]: 
                - 第一个元素表示读取是否成功
                - 第二个元素是捕获的图像帧，如果失败则为None
        """
        if self.cap is None:
            return False, None
            
        # 读取一帧
        ret, frame = self.cap.read()
        
        # 检查读取是否成功
        if not ret:
            return False, None
            
        # 根据配置决定是否水平翻转图像
        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)
            
        return True, frame
    
    def release(self) -> None:
        """
        释放相机资源
        
        关闭相机连接并释放相关资源。应当在程序结束前调用此方法
        以确保相机资源被正确释放。
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None 