import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import yaml
import cv2
import time
import os
import math
from pathlib import Path

from src.ui_utils import UIDrawer

# 引入MediaPipe的手势识别器
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class GestureAnalyzer:
    """
    手势分析类，负责手势识别和状态判断
    
    此类使用MediaPipe库实现手部姿势的检测与跟踪，包括基本的手掌开合状态识别和
    特殊手势（如V手势、拇指向上/向下手势）的检测。支持手势动作触发和轨迹跟踪。
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
        
        # 初始化拇指手势相关变量
        self.is_thumb_up = False
        self.is_thumb_down = False
        self.thumb_cooldown = 0
        self.thumb_cooldown_frames = self.config.get('thumb_gesture_cooldown', 15)  # 拇指手势冷却帧数
        
        # 拇指手势识别参数
        self.thumb_config = self.config.get('thumb_gesture', {})
        self.thumb_straightness_threshold = self.thumb_config.get('straightness_threshold', 0.7)
        self.thumb_bent_fingers_count = self.thumb_config.get('bent_fingers_count', 3)
        self.thumb_y_distance_threshold = self.thumb_config.get('y_distance_threshold', 0.05)
        self.fist_distinction_factor = self.thumb_config.get('fist_distinction_factor', 0.3)
        
        # 初始化MediaPipe手势识别器（如果配置中启用）
        self.use_mediapipe_gesture = self.config.get('use_mediapipe_gesture', True)
        self.gesture_recognizer = None
        self.temp_model_path = None
        
        try:
            if not self.use_mediapipe_gesture:
                return
                
            # 关闭MediaPipe手势识别，使用我们自己的识别逻辑
            if self.config.get('console_output', {}).get('show_startup_info', True):
                print("由于兼容性问题，关闭MediaPipe手势识别，使用内置识别逻辑")
            self.use_mediapipe_gesture = False
            
        except Exception as e:
            if self.config.get('console_output', {}).get('show_error_messages', True):
                print(f"初始化MediaPipe手势识别器时出现错误: {e}")
            self.use_mediapipe_gesture = False
        
        # 设置手势识别结果映射（将MediaPipe手势名称映射到我们的状态名称）
        self.gesture_mapping = {
            "Thumb_Up": "thumb-up",
            "Thumb_Down": "thumb-down",
            "Victory": "v-sign",
            "Closed_Fist": "fist",
            "Open_Palm": "open"
        }
    
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
        min_openness = 0.2  # 最小开合度（握拳状态）
        max_openness = 1.1  # 最大开合度（完全张开状态）
        
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
    
    def _is_thumb_gesture(self, landmarks) -> Tuple[bool, bool]:
        """
        检测拇指向上或向下手势（增强版）
        
        通过分析拇指与其他手指的相对位置以及拇指的朝向来识别拇指向上/向下手势。
        使用多种特征进行判断，增强与握拳状态的区分度。
        
        Args:
            landmarks: 手部关键点列表
            
        Returns:
            Tuple[bool, bool]: (是否拇指向上, 是否拇指向下)
        """
        if not landmarks:
            return False, False
            
        try:
            # 获取调试标志
            show_debug = self.config.get('console_output', {}).get('show_thumb_debug', False)
            
            # 获取拇指指尖、第一关节和食指第一关节的坐标
            # 注意：landmarks可能是MediaPipe原始landmark对象列表，也可能是元组列表
            if hasattr(landmarks[0], 'x'):
                # MediaPipe原始landmark对象
                thumb_tip = landmarks[4]  # 拇指尖
                thumb_ip = landmarks[3]   # 拇指第一关节
                thumb_mcp = landmarks[2]  # 拇指掌指关节
                index_tip = landmarks[8]  # 食指尖
                index_pip = landmarks[6]  # 食指第二关节
                index_mcp = landmarks[5]  # 食指掌指关节
                middle_tip = landmarks[12] # 中指尖
                ring_tip = landmarks[16]   # 无名指尖
                pinky_tip = landmarks[20]  # 小指尖
                wrist = landmarks[0]      # 手腕
                
                # 特征1: 拇指伸直度 - 拇指尖到第一关节的距离与第一关节到掌指关节的距离比较
                thumb_straightness = math.sqrt((thumb_tip.x - thumb_ip.x)**2 + 
                                              (thumb_tip.y - thumb_ip.y)**2 + 
                                              (thumb_tip.z - thumb_ip.z)**2)
                thumb_base_length = math.sqrt((thumb_mcp.x - thumb_ip.x)**2 + 
                                             (thumb_mcp.y - thumb_ip.y)**2 + 
                                             (thumb_mcp.z - thumb_ip.z)**2)
                
                # 特征2: 其他手指弯曲程度 - 检查各指尖与掌指关节的y坐标差异
                finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
                finger_mcps = [index_mcp, landmarks[9], landmarks[13], landmarks[17]]  # 各手指掌指关节
                
                # 计算各手指弯曲程度
                bent_fingers = 0
                finger_bend_ratio = 0  # 用于记录弯曲程度
                finger_bend_details = []  # 记录各手指弯曲详情，用于调试
                
                for i, (tip, mcp) in enumerate(zip(finger_tips, finger_mcps)):
                    # 计算指尖与掌指关节的y坐标差值（正值表示弯曲）
                    bend_amount = tip.y - mcp.y
                    finger_name = ["食指", "中指", "无名指", "小指"][i]
                    finger_bend_details.append(f"{finger_name}: {bend_amount:.3f}")
                    
                    # 弯曲判断
                    if bend_amount > 0:
                        bent_fingers += 1
                        finger_bend_ratio += bend_amount
                
                # 特征3: 拇指与其他手指的分离度 - 拇指尖与其他指尖间的距离
                dist_to_index = math.sqrt((thumb_tip.x - index_tip.x)**2 + 
                                          (thumb_tip.y - index_tip.y)**2 + 
                                          (thumb_tip.z - index_tip.z)**2)
                
                # 计算拇指伸直程度和弯曲手指平均比率
                is_thumb_straight = thumb_straightness > thumb_base_length * self.thumb_straightness_threshold
                
                if bent_fingers > 0:
                    finger_bend_ratio /= bent_fingers  # 平均弯曲度
                    
                # 特征4: 拇指方向 - 拇指尖与掌指关节的y坐标差值
                y_distance = thumb_tip.y - thumb_mcp.y
                
                # 特征5: 拇指朝向 - 使用拇指尖和第一关节的方向向量
                thumb_direction_y = thumb_tip.y - thumb_ip.y  # 负值表示向上，正值表示向下
                
                # 判断是否是握拳状态 - 要求至少指定数量的手指弯曲
                is_fist = bent_fingers >= self.thumb_bent_fingers_count
                
                # 增强特征: 拇指必须明显地与其他手指分开
                is_thumb_separated = dist_to_index > 0.1  # 拇指与食指的距离必须足够大
                
                # 计算区分度
                distinction_ratio = abs(y_distance) / (finger_bend_ratio + 0.001)  # 避免除以零
                
                # 初始化结果
                is_thumb_up = False
                is_thumb_down = False
                
                # 增强判断条件 - 同时考虑多个特征
                if is_thumb_straight and is_fist and is_thumb_separated:
                    if y_distance < -self.thumb_y_distance_threshold and thumb_direction_y < 0:  # 拇指向上
                        if distinction_ratio > self.fist_distinction_factor:
                            is_thumb_up = True
                    elif y_distance > self.thumb_y_distance_threshold and thumb_direction_y > 0:  # 拇指向下
                        if distinction_ratio > self.fist_distinction_factor:
                            is_thumb_down = True
                
                # 详细调试输出
                if show_debug:
                    print(f"拇指手势分析:")
                    print(f"  拇指伸直度: {is_thumb_straight} ({thumb_straightness:.3f} > {thumb_base_length * self.thumb_straightness_threshold:.3f})")
                    print(f"  弯曲手指数: {bent_fingers}/{len(finger_tips)}")
                    print(f"  手指弯曲详情: {', '.join(finger_bend_details)}")
                    print(f"  拇指Y距离: {y_distance:.3f} (阈值: {self.thumb_y_distance_threshold})")
                    print(f"  拇指方向Y: {thumb_direction_y:.3f}")
                    print(f"  拇指分离度: {is_thumb_separated} ({dist_to_index:.3f})")
                    print(f"  区分比率: {distinction_ratio:.3f} (阈值: {self.fist_distinction_factor})")
                    print(f"  判断结果: {'拇指向上' if is_thumb_up else '拇指向下' if is_thumb_down else '非拇指手势'}")
                
            else:
                # landmarks是元组列表，格式为[(x, y, z), ...]
                thumb_tip = np.array(landmarks[4])
                thumb_ip = np.array(landmarks[3])
                thumb_mcp = np.array(landmarks[2])
                index_tip = np.array(landmarks[8])
                index_mcp = np.array(landmarks[5])
                middle_tip = np.array(landmarks[12])
                ring_tip = np.array(landmarks[16])
                pinky_tip = np.array(landmarks[20])
                wrist = np.array(landmarks[0])
                
                # 特征1: 拇指伸直度
                thumb_straightness = np.linalg.norm(thumb_tip - thumb_ip)
                thumb_base_length = np.linalg.norm(thumb_mcp - thumb_ip)
                
                # 特征2: 其他手指弯曲程度
                finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
                finger_mcps = [index_mcp, np.array(landmarks[9]), np.array(landmarks[13]), np.array(landmarks[17])]
                
                # 计算各手指弯曲程度
                bent_fingers = 0
                finger_bend_ratio = 0
                finger_bend_details = []
                
                for i, (tip, mcp) in enumerate(zip(finger_tips, finger_mcps)):
                    bend_amount = tip[1] - mcp[1]  # y坐标差值
                    finger_name = ["食指", "中指", "无名指", "小指"][i]
                    finger_bend_details.append(f"{finger_name}: {bend_amount:.3f}")
                    
                    if bend_amount > 0:
                        bent_fingers += 1
                        finger_bend_ratio += bend_amount
                
                # 特征3: 拇指与其他手指的分离度
                dist_to_index = np.linalg.norm(thumb_tip - index_tip)
                
                # 计算拇指伸直程度和弯曲手指比率
                is_thumb_straight = thumb_straightness > thumb_base_length * self.thumb_straightness_threshold
                
                if bent_fingers > 0:
                    finger_bend_ratio /= bent_fingers
                
                # 特征4: 拇指方向
                y_distance = thumb_tip[1] - thumb_mcp[1]
                
                # 特征5: 拇指朝向
                thumb_direction_y = thumb_tip[1] - thumb_ip[1]
                
                # 判断是否是握拳状态
                is_fist = bent_fingers >= self.thumb_bent_fingers_count
                
                # 拇指分离度
                is_thumb_separated = dist_to_index > 0.1
                
                # 计算区分度
                distinction_ratio = abs(y_distance) / (finger_bend_ratio + 0.001)
                
                # 初始化结果
                is_thumb_up = False
                is_thumb_down = False
                
                # 增强判断条件
                if is_thumb_straight and is_fist and is_thumb_separated:
                    if y_distance < -self.thumb_y_distance_threshold and thumb_direction_y < 0:
                        if distinction_ratio > self.fist_distinction_factor:
                            is_thumb_up = True
                    elif y_distance > self.thumb_y_distance_threshold and thumb_direction_y > 0:
                        if distinction_ratio > self.fist_distinction_factor:
                            is_thumb_down = True
                
                # 详细调试输出
                if show_debug:
                    print(f"拇指手势分析:")
                    print(f"  拇指伸直度: {is_thumb_straight} ({thumb_straightness:.3f} > {thumb_base_length * self.thumb_straightness_threshold:.3f})")
                    print(f"  弯曲手指数: {bent_fingers}/{len(finger_tips)}")
                    print(f"  手指弯曲详情: {', '.join(finger_bend_details)}")
                    print(f"  拇指Y距离: {y_distance:.3f} (阈值: {self.thumb_y_distance_threshold})")
                    print(f"  拇指方向Y: {thumb_direction_y:.3f}")
                    print(f"  拇指分离度: {is_thumb_separated} ({dist_to_index:.3f})")
                    print(f"  区分比率: {distinction_ratio:.3f} (阈值: {self.fist_distinction_factor})")
                    print(f"  判断结果: {'拇指向上' if is_thumb_up else '拇指向下' if is_thumb_down else '非拇指手势'}")
            
            # 检测结果输出
            if is_thumb_up or is_thumb_down:
                console_config = self.config.get('console_output', {})
                if console_config.get('show_vsign_events', True):
                    print(f"检测到拇指手势: {'向上' if is_thumb_up else '向下'}")
            
            return is_thumb_up, is_thumb_down
            
        except Exception as e:
            # 输出错误信息
            console_config = self.config.get('console_output', {})
            if console_config.get('show_error_messages', True):
                print(f"拇指手势检测错误: {e}")
            return False, False
    
    # 使用MediaPipe手势识别器识别手势
    def _recognize_gesture_with_mediapipe(self, frame) -> Optional[str]:
        """
        使用MediaPipe手势识别器识别手势
        
        将输入帧传递给MediaPipe手势识别器，获取识别结果并映射到系统状态
        
        Args:
            frame: 输入图像帧，BGR格式
            
        Returns:
            Optional[str]: 识别的手势名称，如果未识别到则返回None
        """
        if not self.use_mediapipe_gesture or self.gesture_recognizer is None:
            return None
            
        try:
            # 将OpenCV的BGR格式转换为MediaPipe需要的格式
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # 运行手势识别
            recognition_result = self.gesture_recognizer.recognize(mp_image)
            
            # 检查是否有识别结果
            if recognition_result and recognition_result.gestures and len(recognition_result.gestures) > 0:
                # 获取置信度最高的手势
                top_gesture = recognition_result.gestures[0]
                
                if top_gesture and len(top_gesture) > 0:
                    gesture_name = top_gesture[0].category_name
                    confidence = top_gesture[0].score
                    
                    # 检查置信度是否超过阈值
                    gesture_confidence_threshold = self.config.get('gesture_confidence_threshold', 0.7)
                    
                    if confidence >= gesture_confidence_threshold:
                        # 将MediaPipe手势名称映射到我们的状态名称
                        if gesture_name in self.gesture_mapping:
                            mapped_gesture = self.gesture_mapping[gesture_name]
                            
                            # 输出调试信息
                            if self.config.get('console_output', {}).get('show_gesture_recognition', False):
                                print(f"MediaPipe识别手势: {gesture_name}, 映射为: {mapped_gesture}, 置信度: {confidence:.2f}")
                                
                            return mapped_gesture
                        else:
                            # 输出调试信息
                            if self.config.get('console_output', {}).get('show_gesture_recognition', False):
                                print(f"MediaPipe识别到未映射手势: {gesture_name}, 置信度: {confidence:.2f}")
            
            return None
        except Exception as e:
            if self.config.get('console_output', {}).get('show_error_messages', True):
                print(f"MediaPipe手势识别错误: {e}")
            return None
    
    def analyze_frame(self, frame: np.ndarray) -> Tuple[Optional[str], float, bool, Dict[str, Any]]:
        """
        分析单帧图像中的手势
        
        处理输入图像帧，检测手部姿势并分析手势状态。支持基本手势（张开/握拳/V-sign）的
        状态识别和V手势滑动检测。结合防抖机制提高识别稳定性。
        
        Args:
            frame: 输入图像帧，OpenCV格式的numpy数组
            
        Returns:
            Tuple[Optional[str], float, bool, Dict[str, Any]]: 
                - 手势状态: 'open'表示张开，'fist'表示握拳，'v-sign'表示V手势，
                          'thumb-up'表示拇指向上，'thumb-down'表示拇指向下，None表示不确定
                - 开合度: 0.0-1.0的浮点数，表示手掌开合程度
                - 是否是状态转换: 布尔值，指示是否从张开变为握拳的瞬间
                - 额外手势信息: 包含V手势、滑动方向等额外信息的字典
        """
        try:
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
                "swipe_complete": self.swipe_complete,
                "thumb_up": False,
                "thumb_down": False
            }
            
            if not self.last_results.multi_hand_landmarks:
                return None, 0.0, False, extra_gestures
                
            # 获取第一个检测到的手
            hand_landmarks = self.last_results.multi_hand_landmarks[0]
            
            # 提取关键点坐标
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            
            # 计算开合度
            openness = self._calculate_openness(landmarks)
            
            # 使用MediaPipe手势识别器（如果启用）
            mediapipe_gesture = None
            if self.use_mediapipe_gesture:
                mediapipe_gesture = self._recognize_gesture_with_mediapipe(frame)
            
            # 检测V-sign手势
            is_vsign = self._is_vsign_gesture(landmarks)
            extra_gestures["vsign_detected"] = is_vsign
            
            # 检测拇指手势（优先于基本状态检测，避免状态冲突）
            is_thumb_up = False
            is_thumb_down = False
            
            try:
                if self.thumb_cooldown <= 0:
                    # 如果使用MediaPipe手势识别器且识别到了拇指手势，则优先使用其结果
                    if mediapipe_gesture in ['thumb-up', 'thumb-down']:
                        is_thumb_up = mediapipe_gesture == 'thumb-up'
                        is_thumb_down = mediapipe_gesture == 'thumb-down'
                    else:
                        # 否则使用我们自己的拇指手势识别方法
                        is_thumb_up, is_thumb_down = self._is_thumb_gesture(hand_landmarks.landmark)
                    
                    if is_thumb_up or is_thumb_down:
                        self.thumb_cooldown = self.thumb_cooldown_frames
                        extra_gestures["thumb_up"] = is_thumb_up
                        extra_gestures["thumb_down"] = is_thumb_down
                else:
                    self.thumb_cooldown -= 1
            except Exception as e:
                console_config = self.config.get('console_output', {})
                if console_config.get('show_error_messages', True):
                    print(f"处理拇指手势时出错: {e}")
                self.thumb_cooldown -= 1
            
            # 判断基本状态 - 考虑特殊手势的优先级
            thresholds = self.config['thresholds']
            
            # 如果MediaPipe识别到了手势，优先使用其结果
            if mediapipe_gesture:
                new_state = mediapipe_gesture
                
                # 对于V手势，我们仍需要处理滑动相关逻辑
                if new_state == 'v-sign':
                    # 获取跟踪点
                    tracking_point = self._get_vsign_tracking_point(landmarks)
                    extra_gestures["tracking_point"] = tracking_point
                    
                    # 检测V-sign的滑动
                    swiped, direction = self._detect_vsign_swipe(landmarks, frame_width)
                    if swiped:
                        extra_gestures["vsign_swiped"] = True
                        extra_gestures["swipe_direction"] = direction
            else:
                # 否则使用我们自己的手势识别逻辑
                if is_vsign:
                    new_state = 'v-sign'
                    # 获取跟踪点
                    tracking_point = self._get_vsign_tracking_point(landmarks)
                    extra_gestures["tracking_point"] = tracking_point
                    
                    # 检测V-sign的滑动
                    swiped, direction = self._detect_vsign_swipe(landmarks, frame_width)
                    if swiped:
                        extra_gestures["vsign_swiped"] = True
                        extra_gestures["swipe_direction"] = direction
                elif is_thumb_up:
                    new_state = 'thumb-up'
                elif is_thumb_down:
                    new_state = 'thumb-down'
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
            
        except Exception as e:
            console_config = self.config.get('console_output', {})
            if console_config.get('show_error_messages', True):
                print(f"手势分析过程中出错: {e}")
            return None, 0.0, False, {"vsign_detected": False, "vsign_swiped": False, "swipe_direction": None,
                                     "tracking_point": None, "direction_locked": False, "swipe_complete": False,
                                     "thumb_up": False, "thumb_down": False}
    
    @property
    def console_config(self):
        """获取控制台配置"""
        return self.config.get('console_output', {})