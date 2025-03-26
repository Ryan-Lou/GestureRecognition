import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, List
import yaml
import cv2

class GestureAnalyzer:
    """手势分析类，负责手势识别和状态判断"""
    
    def __init__(self, config_path: str):
        """
        初始化手势分析器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.state_counter = 0
        self.current_state = None
        self.last_state = None
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _calculate_openness(self, landmarks: List[Tuple[float, float, float]]) -> float:
        """
        计算手掌开合度
        
        Args:
            landmarks: 手部关键点列表
            
        Returns:
            float: 归一化的开合度值
        """
        if not landmarks:
            return 0.0
            
        # 获取手腕点
        wrist = np.array(landmarks[0])
        
        # 获取四个指尖
        fingertips = [
            np.array(landmarks[8]),  # 食指
            np.array(landmarks[12]), # 中指
            np.array(landmarks[16]), # 无名指
            np.array(landmarks[20])  # 小指
        ]
        
        # 计算平均距离
        distances = [np.linalg.norm(tip - wrist) for tip in fingertips]
        avg_distance = np.mean(distances)
        
        # 归一化处理
        max_distance = np.sqrt(2)  # 最大可能距离（对角线）
        return min(avg_distance / max_distance, 1.0)
    
    def analyze_frame(self, frame: np.ndarray) -> Tuple[Optional[str], float]:
        """
        分析单帧图像中的手势
        
        Args:
            frame: 输入图像帧
            
        Returns:
            Tuple[Optional[str], float]: (手势状态, 开合度)
        """
        # 转换颜色空间
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 处理图像
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None, 0.0
            
        # 获取第一个检测到的手
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # 计算开合度
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        openness = self._calculate_openness(landmarks)
        
        # 判断状态
        thresholds = self.config['thresholds']
        if openness <= thresholds['fist_max']:
            new_state = 'fist'
        elif openness >= thresholds['open_min']:
            new_state = 'open'
        else:
            new_state = None
            
        # 状态防抖
        if new_state == self.current_state:
            self.state_counter += 1
        else:
            self.state_counter = 1
            self.last_state = self.current_state
            self.current_state = new_state
            
        # 需要连续5帧相同状态才确认
        if self.state_counter >= 5:
            return self.current_state, openness
            
        return None, openness
    
    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """
        在图像上绘制手部关键点
        
        Args:
            frame: 输入图像帧
            results: MediaPipe处理结果
            
        Returns:
            np.ndarray: 绘制了关键点的图像帧
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        return frame 