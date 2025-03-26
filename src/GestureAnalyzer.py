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
        self.last_results = None
        self.last_confirmed_state = None  # 添加最后确认的状态
        self.transition_detected = False  # 添加状态转换标记
        
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
        
        # 获取对应的指根点
        finger_bases = [
            np.array(landmarks[5]),  # 食指根
            np.array(landmarks[9]),  # 中指根
            np.array(landmarks[13]), # 无名指根
            np.array(landmarks[17])  # 小指根
        ]
        
        # 计算每个手指的开合度
        finger_openness = []
        for tip, base in zip(fingertips, finger_bases):
            # 计算指尖到指根的距离
            tip_to_base = np.linalg.norm(tip - base)
            # 计算指根到手腕的距离
            base_to_wrist = np.linalg.norm(base - wrist)
            # 计算开合度（指尖到指根的距离 / 指根到手腕的距离）
            openness = tip_to_base / base_to_wrist
            finger_openness.append(openness)
        
        # 计算平均开合度
        avg_openness = np.mean(finger_openness)
        
        # 归一化处理（根据实际测试调整范围）
        min_openness = 0.3  # 最小开合度（握拳状态）
        max_openness = 1.2  # 最大开合度（完全张开状态）
        
        # 将开合度映射到0-1范围
        normalized_openness = (avg_openness - min_openness) / (max_openness - min_openness)
        return max(0.0, min(1.0, normalized_openness))
    
    def analyze_frame(self, frame: np.ndarray) -> Tuple[Optional[str], float, bool]:
        """
        分析单帧图像中的手势
        
        Args:
            frame: 输入图像帧
            
        Returns:
            Tuple[Optional[str], float, bool]: (手势状态, 开合度, 是否是状态转换)
        """
        # 转换颜色空间
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 处理图像
        self.last_results = self.hands.process(rgb_frame)
        
        # 重置转换标记
        self.transition_detected = False
        
        if not self.last_results.multi_hand_landmarks:
            return None, 0.0, False
            
        # 获取第一个检测到的手
        hand_landmarks = self.last_results.multi_hand_landmarks[0]
        
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
            self.current_state = new_state
            
        # 需要连续3帧相同状态才确认
        required_frames = 3  # 减少确认帧数，提高灵敏度
        
        if self.state_counter >= required_frames:
            # 检测状态转换的瞬间
            if self.last_confirmed_state != self.current_state:
                # 记录最后确认的状态
                prev_state = self.last_confirmed_state
                self.last_confirmed_state = self.current_state
                
                # 只有当从open到fist状态转换时设置last_state
                if prev_state == 'open' and self.current_state == 'fist':
                    self.last_state = prev_state
                    self.transition_detected = True
                    print(f"状态变化: {prev_state} -> {self.current_state} (触发)")
                else:
                    print(f"状态变化: {prev_state} -> {self.current_state} (不触发)")
                
                return self.current_state, openness, self.transition_detected
                
            return self.current_state, openness, False
            
        return None, openness, False
    
    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """
        在图像上绘制手部关键点
        
        Args:
            frame: 输入图像帧
            
        Returns:
            np.ndarray: 绘制了关键点的图像帧
        """
        if self.last_results and self.last_results.multi_hand_landmarks:
            for hand_landmarks in self.last_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        return frame