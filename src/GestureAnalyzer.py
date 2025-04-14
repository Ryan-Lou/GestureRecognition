import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import yaml
import cv2
import time
from PIL import Image, ImageDraw, ImageFont
import os

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
    
    def _is_vsign_gesture(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """
        检测是否是V-sign手势（食指与中指伸直并拢，其余手指收回）
        
        Args:
            landmarks: 手部关键点列表
            
        Returns:
            bool: 是否是V-sign手势
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
        获取V-sign手势的跟踪点（食指和中指指尖的中点）
        
        Args:
            landmarks: 手部关键点列表
            
        Returns:
            np.ndarray: 跟踪点坐标
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
        
        Args:
            landmarks: 手部关键点列表
            frame_width: 图像宽度
            
        Returns:
            Tuple[bool, Optional[str]]: (是否触发滑动, 滑动方向)
        """
        if not landmarks:
            return False, None
            
        # 检测是否是V-sign手势
        is_vsign = self._is_vsign_gesture(landmarks)
        
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
        
        # 检测手势状态变化
        if is_vsign:
            # 首次检测到V-sign手势
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
        else:
            # 如果不再是V-sign手势，重置所有状态
            if self.vsign_detected:
                self.vsign_detected = False
                self.vsign_direction = None
                self.direction_locked = False
                self.swipe_complete = False
                
        return triggered, direction
    
    def analyze_frame(self, frame: np.ndarray) -> Tuple[Optional[str], float, bool, Dict[str, Any]]:
        """
        分析单帧图像中的手势
        
        Args:
            frame: 输入图像帧
            
        Returns:
            Tuple[Optional[str], float, bool, Dict[str, Any]]: 
                (手势状态, 开合度, 是否是状态转换, 额外手势信息)
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
        
        # 判断基本状态
        thresholds = self.config['thresholds']
        if openness <= thresholds['fist_max']:
            new_state = 'fist'
        elif openness >= thresholds['open_min']:
            new_state = 'open'
        else:
            new_state = None
        
        # 检测V-sign手势和滑动
        is_vsign = self._is_vsign_gesture(landmarks)
        extra_gestures["vsign_detected"] = is_vsign
        
        # 如果是V-sign手势，获取跟踪点并检测滑动
        if is_vsign:
            tracking_point = self._get_vsign_tracking_point(landmarks)
            extra_gestures["tracking_point"] = tracking_point
            
            swiped, direction = self._detect_vsign_swipe(landmarks, frame_width)
            if swiped:
                extra_gestures["vsign_swiped"] = True
                extra_gestures["swipe_direction"] = direction
        
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
                    # 输出终端信息（根据配置决定是否显示）
                    if self.config.get('console_output', {}).get('show_state_changes', True):
                        print(f"状态变化: {prev_state} -> {self.current_state} (触发)")
                else:
                    # 输出终端信息（根据配置决定是否显示）
                    if self.config.get('console_output', {}).get('show_state_changes', True):
                        print(f"状态变化: {prev_state} -> {self.current_state} (不触发)")
                
                return self.current_state, openness, self.transition_detected, extra_gestures
                
            return self.current_state, openness, False, extra_gestures
            
        return None, openness, False, extra_gestures
    
    def _cv2_put_chinese_text(self, img, text, position, color, font_size=30):
        """
        在OpenCV图像上绘制中文文本
        
        Args:
            img: OpenCV图像
            text: 要绘制的文本
            position: 文本位置，元组(x, y)
            color: 文本颜色，元组(B, G, R)
            font_size: 字体大小
            
        Returns:
            img: 绘制了文本的图像
        """
        # 判断是否是中文
        if any('\u4e00' <= ch <= '\u9fff' for ch in text):
            # 转换图像格式
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # 尝试加载系统中的中文字体
            fontpath = "C:/Windows/Fonts/simhei.ttf"  # 默认黑体
            if not os.path.exists(fontpath):
                # 尝试其他常见字体
                font_options = [
                    "C:/Windows/Fonts/simsun.ttc",    # 宋体
                    "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
                    "C:/Windows/Fonts/simkai.ttf"     # 楷体
                ]
                
                for font_path in font_options:
                    if os.path.exists(font_path):
                        fontpath = font_path
                        break
            
            # 加载字体
            try:
                font = ImageFont.truetype(fontpath, font_size)
            except IOError:
                # 如果无法加载字体，使用默认字体
                font = ImageFont.load_default()
            
            # 绘制文本
            draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
            
            # 转换回OpenCV格式
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        else:
            # 如果不是中文，使用OpenCV原生方法
            return cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size/60, color, 2)
    
    def draw_landmarks(self, frame: np.ndarray, extra_info: Dict[str, Any] = None) -> np.ndarray:
        """
        在图像上绘制手部关键点和额外信息
        
        Args:
            frame: 输入图像帧
            extra_info: 额外信息字典
            
        Returns:
            np.ndarray: 绘制了关键点的图像帧
        """
        if self.last_results and self.last_results.multi_hand_landmarks:
            height, width = frame.shape[:2]
            
            # 获取可视化配置
            viz_config = self.config.get('visualization', {})
            
            # UI元素位置和样式参数（可在此处调整）
            vsign_font_size = 30                  # V手势信息字体大小
            vsign_position = [150, 60]            # V手势信息位置 [x偏移量（从右边开始算）, y]
            direction_position = [10, 60]         # 方向信息位置 [x, y]
            lock_position = [10, 90]              # 锁定状态位置 [x, y]
            reset_position = [10, 120]            # 复位状态位置 [x, y]
            
            # 绘制手部关键点和连接线
            if viz_config.get('show_landmarks', True):
                for hand_landmarks in self.last_results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    
                    # 获取关键点列表
                    landmarks = hand_landmarks.landmark
                    
                    # 获取食指和中指指尖位置
                    index_tip = (int(landmarks[8].x * width), int(landmarks[8].y * height))
                    middle_tip = (int(landmarks[12].x * width), int(landmarks[12].y * height))
                    
                    # 计算指尖中点位置（跟踪点）
                    tracking_point = (
                        (index_tip[0] + middle_tip[0]) // 2,
                        (index_tip[1] + middle_tip[1]) // 2
                    )
                    
                    # 绘制指尖标记
                    cv2.circle(frame, index_tip, 8, (0, 255, 255), -1)
                    cv2.circle(frame, middle_tip, 8, (0, 255, 255), -1)
                    
                    # 绘制跟踪点（较大圆圈）
                    cv2.circle(frame, tracking_point, 12, (255, 0, 255), -1)
                    
                    # 如果有V手势滑动，显示相关信息
                    if viz_config.get('show_vsign_info', True):
                        # 滑动方向信息
                        if extra_info.get("vsign_swiped", False):
                            direction = extra_info.get("swipe_direction")
                            if direction:
                                direction_text = f"滑动: {'右' if direction == 'right' else '左'}"
                                frame = self._cv2_put_chinese_text(
                                    frame, 
                                    direction_text, 
                                    (direction_position[0], direction_position[1]), 
                                    (255, 0, 255), 
                                    vsign_font_size
                                )
                        
                        # V手势类型提示
                        frame = self._cv2_put_chinese_text(
                            frame, 
                            "V手势", 
                            (width - vsign_position[0], vsign_position[1]), 
                            (255, 0, 255), 
                            vsign_font_size
                        )
                        
                        # 方向锁定状态
                        lock_status = "已锁定" if self.direction_locked else "未锁定"
                        lock_color = (0, 0, 255) if self.direction_locked else (0, 255, 0)
                        frame = self._cv2_put_chinese_text(
                            frame, 
                            f"方向: {lock_status}", 
                            (lock_position[0], lock_position[1]), 
                            lock_color, 
                            25
                        )
                        
                        # 显示复位状态
                        reset_text = ""
                        if self.direction_locked and self.swipe_complete:
                            # 计算当前位置与屏幕中心的距离
                            if extra_info and extra_info.get("tracking_point") is not None:
                                current_pos = extra_info["tracking_point"]
                                current_x = current_pos[0]  # 归一化的x坐标 (0-1)
                                distance_to_center = abs(current_x - self.center_position)
                                is_resetting = distance_to_center < self.reset_threshold
                                
                                if is_resetting:
                                    reset_text = "即将完成复位"
                                else:
                                    reset_text = "请将手移回中心位置"
                        
                        if reset_text:
                            frame = self._cv2_put_chinese_text(
                                frame, 
                                reset_text, 
                                (reset_position[0], reset_position[1]), 
                                (0, 255, 255), 
                                25
                            )
            
            # 绘制中心位置标记线
            if viz_config.get('show_center_line', True):
                center_x = int(self.center_position * width)
                cv2.line(frame, (center_x, 0), (center_x, height), (0, 255, 0), 1)
            
            # 绘制复位区域
            if viz_config.get('show_reset_area', True):
                reset_left = int((self.center_position - self.reset_threshold) * width)
                reset_right = int((self.center_position + self.reset_threshold) * width)
                cv2.rectangle(frame, (reset_left, 0), (reset_right, 30), (0, 255, 255), -1)
                frame = self._cv2_put_chinese_text(frame, "复位区域", (reset_left + 10, 20), (0, 0, 0), 20)
                
            # 绘制轨迹
            if viz_config.get('show_tracking_path', True) and len(self.position_history) >= 2:
                for i in range(1, len(self.position_history)):
                    pt1 = (int(self.position_history[i-1][0] * frame.shape[1]), 
                          int(self.position_history[i-1][1] * frame.shape[0]))
                    pt2 = (int(self.position_history[i][0] * frame.shape[1]), 
                          int(self.position_history[i][1] * frame.shape[0]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                
        return frame