import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import yaml
import cv2
import time
import os

from src.ui_utils import UIDrawer

class GestureAnalyzer:
    """
    手势分析类，负责手势识别和状态判断
    
    此类使用MediaPipe库实现手部姿势的检测与跟踪，包括基本的手掌开合状态识别和
    特殊手势（如V手势）的检测。支持手势动作触发和轨迹跟踪。
    """
    
    def __init__(self, config_path: str):
        """
        初始化手势分析器
        
        加载配置文件，初始化MediaPipe手部检测模型和相关状态变量，
        设置手势识别参数和轨迹跟踪机制。
        
        Args:
            config_path: 配置文件路径，YAML格式
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
        
        # V-sign手势相关变量
        self.vsign_detected = False
        self.last_vsign_position = None
        self.vsign_start_time = None
        self.vsign_direction = None
        self.vsign_trigger_threshold = self.config.get('vsign_trigger_threshold', 0.1)  # 默认为屏幕宽度的10%
        self.last_trigger_time = 0
        self.cooldown = self.config['cooldown']
        
        # 手势轨迹跟踪
        self.position_history = []
        self.max_history_length = 10  # 保存最近10帧的位置
        
        # 添加方向锁定和复位相关变量
        self.direction_locked = False  # 方向锁定标记
        self.swipe_complete = False    # 滑动完成标记
        self.center_position = self.config.get('center_position', 0.5)  # 屏幕中心位置（归一化坐标，范围0-1）
        self.reset_threshold = self.config.get('reset_threshold', 0.1)  # 复位阈值，接近中心位置的范围
        
    def _load_config(self, config_path: str) -> dict:
        """
        加载配置文件
        
        从指定路径加载YAML格式的配置文件，用于设置手势识别参数。
        
        Args:
            config_path: 配置文件的路径
            
        Returns:
            dict: 包含配置参数的字典
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _calculate_openness(self, landmarks: List[Tuple[float, float, float]]) -> float:
        """
        计算手掌开合度
        
        基于手部关键点位置计算手掌的开合程度，通过分析四个手指（食指、中指、无名指、小指）
        的指尖到指根距离与指根到手腕距离的比值，得出归一化的开合度值。
        
        Args:
            landmarks: 手部关键点列表，每个关键点为(x, y, z)坐标元组
            
        Returns:
            float: 归一化的开合度值，范围0.0-1.0，0表示完全握拳，1表示完全张开
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
    
    def _is_vsign_gesture(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """
        检测是否是V-sign手势
        
        判断当前手势是否为V手势（食指与中指伸直并拢，其余手指收回）。
        通过分析各手指的伸直程度和食指与中指的并拢程度来判断。
        
        Args:
            landmarks: 手部关键点列表，每个关键点为(x, y, z)坐标元组
            
        Returns:
            bool: 如果检测到V手势则返回True，否则返回False
        """
        if not landmarks:
            return False
            
        # 获取手指关键点
        wrist = np.array(landmarks[0])
        thumb_tip = np.array(landmarks[4])
        index_tip = np.array(landmarks[8])
        middle_tip = np.array(landmarks[12])
        ring_tip = np.array(landmarks[16])
        pinky_tip = np.array(landmarks[20])
        
        # 获取手指中间关节点
        index_pip = np.array(landmarks[6])
        middle_pip = np.array(landmarks[10])
        ring_pip = np.array(landmarks[14])
        pinky_pip = np.array(landmarks[18])
        
        # 获取手指指根点
        index_mcp = np.array(landmarks[5])
        middle_mcp = np.array(landmarks[9])
        
        # 计算各手指伸直程度（指尖到手腕的距离与指根到手腕的距离的比值）
        index_extension = np.linalg.norm(index_tip - wrist) / np.linalg.norm(index_mcp - wrist)
        middle_extension = np.linalg.norm(middle_tip - wrist) / np.linalg.norm(middle_mcp - wrist)
        ring_extension = np.linalg.norm(ring_tip - wrist) / np.linalg.norm(ring_pip - wrist)
        pinky_extension = np.linalg.norm(pinky_tip - wrist) / np.linalg.norm(pinky_pip - wrist)
        
        # 计算食指和中指的并拢程度
        finger_closeness = np.linalg.norm(index_tip - middle_tip)
        
        # 判断V-sign手势条件：
        # 1. 食指和中指伸直（伸直程度高）
        # 2. 其他手指收回（伸直程度低）
        # 3. 食指和中指并拢（距离近）
        threshold_extension_high = 1.3  # 伸直手指的阈值
        threshold_extension_low = 1.1   # 弯曲手指的阈值
        threshold_closeness = 0.1       # 并拢手指的距离阈值
        
        is_vsign = (index_extension > threshold_extension_high and 
                   middle_extension > threshold_extension_high and 
                   ring_extension < threshold_extension_low and 
                   pinky_extension < threshold_extension_low and 
                   finger_closeness < threshold_closeness)
                    
        return is_vsign
    
    def _get_vsign_tracking_point(self, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        获取V-sign手势的跟踪点
        
        计算V手势的代表性跟踪点，定义为食指和中指指尖的中点位置。
        该点用于跟踪V手势的移动轨迹。
        
        Args:
            landmarks: 手部关键点列表，每个关键点为(x, y, z)坐标元组
            
        Returns:
            np.ndarray: 跟踪点的三维坐标，格式为[x, y, z]
        """
        if not landmarks:
            return np.array([0, 0, 0])
            
        # 获取食指和中指指尖
        index_tip = np.array(landmarks[8])
        middle_tip = np.array(landmarks[12])
        
        # 计算指尖中点位置
        tracking_point = (index_tip + middle_tip) / 2
        
        return tracking_point
    
    def _detect_vsign_swipe(self, landmarks: List[Tuple[float, float, float]], frame_width: int) -> Tuple[bool, Optional[str]]:
        """
        检测V-sign手势的滑动
        
        检测并分析V手势的水平滑动动作，用于触发方向控制。
        支持左右方向识别、方向锁定和复位机制，防止误触发。
        
        Args:
            landmarks: 手部关键点列表，每个关键点为(x, y, z)坐标元组
            frame_width: 图像宽度，用于归一化滑动距离
            
        Returns:
            Tuple[bool, Optional[str]]: 返回一个元组，第一个元素表示是否触发滑动，
                                       第二个元素表示滑动方向('left'或'right')，
                                       若未触发则方向为None
        """
        if not landmarks:
            return False, None
            
        # 获取V-sign手势的跟踪点（食指和中指指尖的中点）
        tracking_point = self._get_vsign_tracking_point(landmarks)
        current_position = tracking_point[:2]  # 只取x,y坐标
        
        # 更新位置历史
        self.position_history.append(current_position)
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
        
        # 滑动判断结果
        triggered = False
        direction = None
        
        # 如果是第一次检测到V-sign，初始化位置
        if not self.vsign_detected:
            self.vsign_detected = True
            self.last_vsign_position = current_position
            self.vsign_start_time = time.time()
            self.vsign_direction = None
            self.direction_locked = False  # 重置方向锁定
            self.swipe_complete = False    # 重置滑动完成标记
        else:
            # 计算水平移动距离
            horizontal_movement = current_position[0] - self.last_vsign_position[0]
            normalized_movement = horizontal_movement  # 相对于实际坐标系统的移动（0-1范围）
            
            # 计算移动距离相对于屏幕宽度的比例
            movement_ratio = abs(normalized_movement)
            
            # 计算当前位置与屏幕中心的距离
            distance_to_center = abs(current_position[0] - self.center_position)
            
            # 判断是否在复位过程中（接近屏幕中心）
            is_resetting = distance_to_center < self.reset_threshold
            
            # 检查是否超过触发阈值，且未处于方向锁定状态
            current_time = time.time()
            if (movement_ratio > self.vsign_trigger_threshold and 
                current_time - self.last_trigger_time > self.cooldown and
                not self.direction_locked and not self.swipe_complete):
                
                if horizontal_movement > 0:
                    direction = "right"
                else:
                    direction = "left"
                    
                # 只有当方向确定且与上次不同或已重置时才触发
                if self.vsign_direction is None or self.vsign_direction != direction:
                    self.vsign_direction = direction
                    triggered = True
                    self.last_trigger_time = current_time
                    # 锁定方向，防止复位过程中误触发
                    self.direction_locked = True
                    self.swipe_complete = True  # 标记滑动已完成
                    # 输出终端信息（根据配置决定是否显示）
                    if self.config.get('console_output', {}).get('show_vsign_events', True):
                        print(f"锁定方向: {direction}，等待复位")
                    # 重置起始位置，为下一次滑动做准备
                    self.last_vsign_position = current_position
            
            # 如果已经完成滑动且接近中心位置，解锁方向锁定
            if self.swipe_complete and is_resetting:
                self.direction_locked = False
                self.swipe_complete = False
                self.vsign_direction = None
                # 输出终端信息（根据配置决定是否显示）
                if self.config.get('console_output', {}).get('show_reset_events', True):
                    print("复位完成，解除方向锁定")
                self.last_vsign_position = current_position
                
        return triggered, direction
    
    def analyze_frame(self, frame: np.ndarray) -> Tuple[Optional[str], float, bool, Dict[str, Any]]:
        """
        分析单帧图像中的手势
        
        处理输入图像帧，检测手部姿势并分析手势状态。支持基本手势（张开/握拳/V-sign）的
        状态识别和V手势滑动检测。结合防抖机制提高识别稳定性。
        
        Args:
            frame: 输入图像帧，OpenCV格式的numpy数组
            
        Returns:
            Tuple[Optional[str], float, bool, Dict[str, Any]]: 
                - 手势状态: 'open'表示张开，'fist'表示握拳，'v-sign'表示V手势，None表示不确定
                - 开合度: 0.0-1.0的浮点数，表示手掌开合程度
                - 是否是状态转换: 布尔值，指示是否从张开变为握拳的瞬间
                - 额外手势信息: 包含V手势、滑动方向等额外信息的字典
        """
        frame_height, frame_width = frame.shape[:2]
        
        # 转换颜色空间
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 处理图像
        self.last_results = self.hands.process(rgb_frame)
        
        # 重置转换标记
        self.transition_detected = False
        
        # 额外手势信息
        extra_gestures = {
            "vsign_detected": False,
            "vsign_swiped": False,
            "swipe_direction": None,
            "tracking_point": None,
            "direction_locked": self.direction_locked,
            "swipe_complete": self.swipe_complete
        }
        
        if not self.last_results.multi_hand_landmarks:
            return None, 0.0, False, extra_gestures
            
        # 获取第一个检测到的手
        hand_landmarks = self.last_results.multi_hand_landmarks[0]
        
        # 提取关键点坐标
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        
        # 计算开合度
        openness = self._calculate_openness(landmarks)
        
        # 检测V-sign手势
        is_vsign = self._is_vsign_gesture(landmarks)
        extra_gestures["vsign_detected"] = is_vsign
        
        # 判断基本状态
        thresholds = self.config['thresholds']
        if is_vsign:
            new_state = 'v-sign'
            # 如果是V-sign，获取跟踪点
            tracking_point = self._get_vsign_tracking_point(landmarks)
            extra_gestures["tracking_point"] = tracking_point
            
            # 检测V-sign的滑动
            swiped, direction = self._detect_vsign_swipe(landmarks, frame_width)
            if swiped:
                extra_gestures["vsign_swiped"] = True
                extra_gestures["swipe_direction"] = direction
        elif openness <= thresholds['fist_max']:
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
                
                # 只有当从open到fist状态转换时设置last_state和触发标记
                if prev_state == 'open' and self.current_state == 'fist':
                    self.last_state = prev_state
                    self.transition_detected = True
                    # 输出终端信息（根据配置决定是否显示）
                    if self.config.get('console_output', {}).get('show_state_changes', True):
                        print(f"状态变化: {prev_state} -> {self.current_state} (触发)")
                else:
                    # 其他状态变化不触发，但记录last_state
                    self.last_state = prev_state
                    # 输出终端信息（根据配置决定是否显示）
                    if self.config.get('console_output', {}).get('show_state_changes', True):
                        print(f"状态变化: {prev_state} -> {self.current_state} (不触发)")
                
                return self.current_state, openness, self.transition_detected, extra_gestures
                
            return self.current_state, openness, False, extra_gestures
            
        return None, openness, False, extra_gestures