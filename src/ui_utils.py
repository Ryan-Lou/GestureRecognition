import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Dict, Any, Tuple, Optional

# -------------------------------------------------------------
# UI常量定义 - 集中管理UI元素样式参数
# -------------------------------------------------------------
class UIConfig:
    """
    UI配置常量类
    
    此类集中管理所有UI相关的常量配置，包括尺寸、颜色、位置等参数，
    便于统一修改和维护UI样式。
    """
    
    # 基础配置
    MARGIN = 20                     # 基础边距
    FONT_BASE_SIZE = 24             # 基础字体大小
    FONT_SMALL_SIZE = 18            # 小号字体大小
    FONT_LARGE_SIZE = 36            # 大号字体大小
    
    # 颜色定义
    COLOR_WHITE = (255, 255, 255)   # 白色
    COLOR_BLACK = (0, 0, 0)         # 黑色
    COLOR_GREEN = (0, 255, 0)       # 绿色
    COLOR_RED = (0, 0, 255)         # 红色（OpenCV中为BGR）
    COLOR_BLUE = (255, 0, 0)        # 蓝色
    COLOR_YELLOW = (0, 255, 255)    # 黄色
    COLOR_PURPLE = (255, 0, 255)    # 紫色
    COLOR_CYAN = (255, 255, 0)      # 青色
    
    # 状态文本配置（左上角）
    STATE_POSITION_X = MARGIN
    STATE_POSITION_Y = MARGIN + 5
    STATE_FONT_SIZE = FONT_BASE_SIZE
    
    # FPS显示配置（右上角）
    FPS_POSITION_X = MARGIN
    FPS_POSITION_Y = MARGIN + 5
    FPS_FONT_SIZE = FONT_BASE_SIZE
    
    # 手势说明配置（右侧）
    INSTR_X_OFFSET = MARGIN + 260
    INSTR_Y_START = 70
    INSTR_LINE_SPACING = 30
    INSTR_FONT_SIZE = FONT_SMALL_SIZE
    
    # 进度条配置（底部）
    BAR_WIDTH = 250
    BAR_HEIGHT = 25
    BAR_X_OFFSET = MARGIN
    BAR_Y_OFFSET = MARGIN + 30
    BAR_TEXT_FONT_SIZE = FONT_SMALL_SIZE
    
    # 历史状态配置（底部）
    HISTORY_POSITION_X = MARGIN
    HISTORY_POSITION_Y = BAR_Y_OFFSET + BAR_HEIGHT + 25
    HISTORY_FONT_SIZE = FONT_SMALL_SIZE
    
    # 触发提示配置（画面中央）
    TRIGGER_FONT_SIZE = FONT_LARGE_SIZE
    TRIGGER_Y_OFFSET = 60
    TRIGGER_BG_PADDING = 20
    TRIGGER_BG_ALPHA = 0.5
    TRIGGER_TEXT_COLOR = COLOR_RED
    TRIGGER_DIR_COLOR = COLOR_BLUE
    
    # V手势信息配置
    VSIGN_FONT_SIZE = FONT_BASE_SIZE
    VSIGN_POSITION_X = 180
    VSIGN_POSITION_Y = 70
    DIRECTION_POSITION_X = MARGIN
    DIRECTION_POSITION_Y = 60
    LOCK_POSITION_X = MARGIN
    LOCK_POSITION_Y = 90
    RESET_POSITION_X = MARGIN
    RESET_POSITION_Y = 120
    LOCK_FONT_SIZE = FONT_SMALL_SIZE
    RESET_FONT_SIZE = FONT_SMALL_SIZE
    
    # 手部关键点配置
    INDEX_TIP_SIZE = 8
    MIDDLE_TIP_SIZE = 8
    TRACKING_POINT_SIZE = 12
    
    # 中心线和复位区域
    CENTER_LINE_COLOR = COLOR_GREEN
    CENTER_LINE_THICKNESS = 1
    RESET_AREA_COLOR = COLOR_YELLOW
    RESET_AREA_HEIGHT = 30
    RESET_AREA_TEXT = "复位区域"
    RESET_AREA_FONT_SIZE = 20
    
    # 轨迹配置
    TRACK_COLOR = COLOR_GREEN
    TRACK_THICKNESS = 2


class UIDrawer:
    """
    UI绘制工具类
    
    提供静态方法用于在OpenCV图像上绘制各种UI元素，包括文本、图形和
    交互提示等。此类集中处理所有绘图逻辑，便于维护和扩展。
    """
    
    @staticmethod
    def put_chinese_text(img: np.ndarray, text: str, position: Tuple[int, int], 
                       color: Tuple[int, int, int], font_size: int = 30) -> np.ndarray:
        """
        在OpenCV图像上绘制中文文本
        
        使用PIL库在OpenCV图像上绘制支持中文的文本，解决OpenCV原生
        无法显示中文的问题。自动判断文本是否包含中文，选择合适的渲染方式。
        
        Args:
            img: OpenCV格式的图像数组
            text: 要绘制的文本内容，支持中文
            position: 文本位置坐标，格式为(x, y)
            color: 文本颜色，BGR格式的元组，如(255, 0, 0)表示蓝色
            font_size: 字体大小，默认为30像素
            
        Returns:
            np.ndarray: 绘制了文本的新图像
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
    
    @staticmethod
    def draw_ui_elements(frame: np.ndarray, state: Optional[str], openness: float, 
                        fps: float, analyzer_instance, viz_config: Dict) -> np.ndarray:
        """
        绘制主界面UI元素
        
        在图像上绘制各种UI元素，包括状态文本、FPS计数器、开合度进度条、
        手势历史记录和操作说明等，构成完整的用户界面。
        
        Args:
            frame: 输入图像帧
            state: 当前手势状态，可以是'open'、'fist'、'v-sign'或None
            openness: 手掌开合度，范围为0.0-1.0
            fps: 当前帧率
            analyzer_instance: 手势分析器实例，用于获取历史状态信息
            viz_config: 可视化配置字典，控制各UI元素的显示与隐藏
            
        Returns:
            np.ndarray: 绘制了完整UI的图像帧
        """
        height, width = frame.shape[:2]
        
        # 绘制状态文本（左上角）
        if state and viz_config.get('show_state', True):
            # 根据状态选择颜色
            if state == 'open':
                color = UIConfig.COLOR_GREEN
                state_text = "状态: 张开"
            elif state == 'fist':
                color = UIConfig.COLOR_RED
                state_text = "状态: 握拳"
            elif state == 'v-sign':
                color = UIConfig.COLOR_PURPLE
                state_text = "状态: V手势"
            else:
                color = UIConfig.COLOR_WHITE
                state_text = f"状态: {state}"
                
            frame = UIDrawer.put_chinese_text(
                frame, 
                state_text, 
                (UIConfig.STATE_POSITION_X, UIConfig.STATE_POSITION_Y), 
                color, 
                UIConfig.STATE_FONT_SIZE
            )
        
        # 绘制FPS（右上角）
        if viz_config.get('show_fps', True):
            fps_text = f"FPS: {fps:.1f}"
            text_size = len(fps_text) * (UIConfig.FPS_FONT_SIZE // 2)  # 估算文本宽度
            frame = UIDrawer.put_chinese_text(
                frame, 
                fps_text, 
                (width - UIConfig.FPS_POSITION_X - text_size, UIConfig.FPS_POSITION_Y), 
                UIConfig.COLOR_GREEN, 
                UIConfig.FPS_FONT_SIZE
            )
        
        # 绘制开合度进度条（底部）
        if viz_config.get('show_openness_bar', True):
            # 计算进度条位置
            bar_x = UIConfig.BAR_X_OFFSET
            bar_y = height - UIConfig.BAR_Y_OFFSET
            
            # 背景
            cv2.rectangle(frame, (bar_x, bar_y),
                        (bar_x + UIConfig.BAR_WIDTH, bar_y + UIConfig.BAR_HEIGHT),
                        UIConfig.COLOR_WHITE, -1)
            
            # 进度
            progress_width = int(UIConfig.BAR_WIDTH * openness)
            cv2.rectangle(frame, (bar_x, bar_y),
                        (bar_x + progress_width, bar_y + UIConfig.BAR_HEIGHT),
                        UIConfig.COLOR_GREEN, -1)
            
            # 边框
            cv2.rectangle(frame, (bar_x, bar_y),
                        (bar_x + UIConfig.BAR_WIDTH, bar_y + UIConfig.BAR_HEIGHT),
                        UIConfig.COLOR_BLACK, 2)
            
            # 显示开合度数值
            openness_text = f"开合度: {openness:.2f}"
            frame = UIDrawer.put_chinese_text(
                frame, 
                openness_text, 
                (bar_x + UIConfig.BAR_WIDTH + 10, bar_y + UIConfig.BAR_HEIGHT - 5), 
                UIConfig.COLOR_BLACK, 
                UIConfig.BAR_TEXT_FONT_SIZE
            )
        
        # 显示最新状态（底部）
        if viz_config.get('show_gesture_history', True):
            last_state = analyzer_instance.last_state
            current_state = analyzer_instance.current_state
            if last_state and current_state:
                # 转换状态文本
                if last_state == 'open':
                    last_state_text = '张开'
                elif last_state == 'fist':
                    last_state_text = '握拳'
                elif last_state == 'v-sign':
                    last_state_text = 'V手势'
                else:
                    last_state_text = last_state
                    
                if current_state == 'open':
                    current_state_text = '张开'
                elif current_state == 'fist':
                    current_state_text = '握拳'
                elif current_state == 'v-sign':
                    current_state_text = 'V手势'
                else:
                    current_state_text = current_state
                
                state_text = f"上次: {last_state_text}, 当前: {current_state_text}"
                frame = UIDrawer.put_chinese_text(
                    frame, 
                    state_text, 
                    (UIConfig.HISTORY_POSITION_X, height - UIConfig.HISTORY_POSITION_Y), 
                    UIConfig.COLOR_RED, 
                    UIConfig.HISTORY_FONT_SIZE
                )
            
        # 显示手势说明（右侧）
        if viz_config.get('show_instructions', True):
            instructions = [
                "手势说明:",
                "1. 从张开到握拳: 空格键",
                "2. V手势向右滑动: 前进",
                "3. V手势向左滑动: 后退"
            ]
            
            for i, text in enumerate(instructions):
                pos_y = UIConfig.INSTR_Y_START + i * UIConfig.INSTR_LINE_SPACING
                frame = UIDrawer.put_chinese_text(
                    frame, 
                    text, 
                    (width - UIConfig.INSTR_X_OFFSET, pos_y), 
                    UIConfig.COLOR_BLACK, 
                    UIConfig.INSTR_FONT_SIZE
                )
        
        return frame
    
    @staticmethod
    def draw_trigger_notification(frame: np.ndarray, text: str, y_offset: int = 0, 
                                color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        在画面中央绘制触发提示
        
        在图像中央绘制带有半透明背景的提示文本，用于显示动作触发的反馈。
        支持自定义文本位置偏移和颜色。
        
        Args:
            frame: 输入图像帧
            text: 提示文本内容
            y_offset: Y轴偏移量，用于调整文本垂直位置
            color: 文本颜色，BGR格式，如不指定则使用默认值

        Returns:
            np.ndarray: 绘制了提示的图像帧
        """
        if color is None:
            color = UIConfig.TRIGGER_TEXT_COLOR
            
        # 计算文本位置（画面中央）
        text_size = len(text) * (UIConfig.TRIGGER_FONT_SIZE // 2)  # 估算文本宽度
        text_x = frame.shape[1]//2 - text_size//2
        text_y = frame.shape[0]//2 + y_offset
        
        # 添加半透明背景
        overlay = frame.copy()
        bg_x1 = text_x - UIConfig.TRIGGER_BG_PADDING
        bg_y1 = text_y - UIConfig.TRIGGER_FONT_SIZE - UIConfig.TRIGGER_BG_PADDING
        bg_x2 = text_x + text_size + UIConfig.TRIGGER_BG_PADDING
        bg_y2 = text_y + UIConfig.TRIGGER_BG_PADDING
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), UIConfig.COLOR_BLACK, -1)
        
        # 应用透明度
        frame = cv2.addWeighted(overlay, UIConfig.TRIGGER_BG_ALPHA, frame, 1 - UIConfig.TRIGGER_BG_ALPHA, 0)
        
        # 绘制文本
        frame = UIDrawer.put_chinese_text(
            frame, 
            text, 
            (text_x, text_y), 
            color, 
            UIConfig.TRIGGER_FONT_SIZE
        )
        
        return frame
    
    @staticmethod
    def draw_hand_landmarks(frame: np.ndarray, landmarks_results, extra_info: Dict[str, Any],
                          mp_draw, mp_hands, center_position: float, reset_threshold: float,
                          direction_locked: bool, swipe_complete: bool, position_history: list,
                          viz_config: Dict) -> np.ndarray:
        """
        绘制手部关键点、轨迹和相关信息
        
        在图像上绘制MediaPipe检测到的手部关键点、连接线、跟踪点和手势轨迹，
        并显示V手势相关状态信息和方向指示。
        
        Args:
            frame: 输入图像帧
            landmarks_results: MediaPipe手部关键点检测结果
            extra_info: 额外手势信息字典，包含V手势状态等数据
            mp_draw: MediaPipe绘图工具
            mp_hands: MediaPipe手部模型
            center_position: 中心位置的归一化坐标（0-1范围）
            reset_threshold: 复位阈值，接近中心位置的范围
            direction_locked: 方向是否锁定的标志
            swipe_complete: 滑动是否完成的标志
            position_history: 手势位置历史记录列表
            viz_config: 可视化配置字典，控制各UI元素的显示与隐藏
            
        Returns:
            np.ndarray: 绘制了手部关键点和相关信息的图像帧
        """
        if not landmarks_results or not landmarks_results.multi_hand_landmarks:
            return frame
            
        height, width = frame.shape[:2]
        
        # 绘制手部关键点和连接线
        if viz_config.get('show_landmarks', True):
            for hand_landmarks in landmarks_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=UIConfig.COLOR_GREEN, thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=UIConfig.COLOR_RED, thickness=2)
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
                cv2.circle(frame, index_tip, UIConfig.INDEX_TIP_SIZE, UIConfig.COLOR_YELLOW, -1)
                cv2.circle(frame, middle_tip, UIConfig.MIDDLE_TIP_SIZE, UIConfig.COLOR_YELLOW, -1)
                
                # 绘制跟踪点（较大圆圈）
                cv2.circle(frame, tracking_point, UIConfig.TRACKING_POINT_SIZE, UIConfig.COLOR_PURPLE, -1)
                
                # 如果有V手势滑动，显示相关信息
                if viz_config.get('show_vsign_info', True):
                    # 滑动方向信息
                    if extra_info.get("vsign_swiped", False):
                        direction = extra_info.get("swipe_direction")
                        if direction:
                            direction_text = f"滑动: {'右' if direction == 'right' else '左'}"
                            frame = UIDrawer.put_chinese_text(
                                frame, 
                                direction_text, 
                                (UIConfig.DIRECTION_POSITION_X, UIConfig.DIRECTION_POSITION_Y), 
                                UIConfig.COLOR_PURPLE, 
                                UIConfig.VSIGN_FONT_SIZE
                            )
                    
                    # 方向锁定状态
                    lock_status = "已锁定" if direction_locked else "未锁定"
                    lock_color = UIConfig.COLOR_RED if direction_locked else UIConfig.COLOR_GREEN
                    frame = UIDrawer.put_chinese_text(
                        frame, 
                        f"方向: {lock_status}", 
                        (UIConfig.LOCK_POSITION_X, UIConfig.LOCK_POSITION_Y), 
                        lock_color, 
                        UIConfig.LOCK_FONT_SIZE
                    )
                    
                    # 显示复位状态
                    reset_text = ""
                    if direction_locked and swipe_complete:
                        # 计算当前位置与屏幕中心的距离
                        if extra_info and extra_info.get("tracking_point") is not None:
                            current_pos = extra_info["tracking_point"]
                            current_x = current_pos[0]  # 归一化的x坐标 (0-1)
                            distance_to_center = abs(current_x - center_position)
                            is_resetting = distance_to_center < reset_threshold
                            
                            if is_resetting:
                                reset_text = "即将完成复位"
                            else:
                                reset_text = "请将手移回中心位置"
                    
                    if reset_text:
                        frame = UIDrawer.put_chinese_text(
                            frame, 
                            reset_text, 
                            (UIConfig.RESET_POSITION_X, UIConfig.RESET_POSITION_Y), 
                            UIConfig.COLOR_YELLOW, 
                            UIConfig.RESET_FONT_SIZE
                        )
        
        # 绘制中心位置标记线
        if viz_config.get('show_center_line', True):
            center_x = int(center_position * width)
            cv2.line(frame, (center_x, 0), (center_x, height), 
                    UIConfig.CENTER_LINE_COLOR, UIConfig.CENTER_LINE_THICKNESS)
        
        # 绘制复位区域
        if viz_config.get('show_reset_area', True):
            reset_left = int((center_position - reset_threshold) * width)
            reset_right = int((center_position + reset_threshold) * width)
            cv2.rectangle(frame, (reset_left, 0), (reset_right, UIConfig.RESET_AREA_HEIGHT), 
                        UIConfig.RESET_AREA_COLOR, -1)
            frame = UIDrawer.put_chinese_text(
                frame, 
                UIConfig.RESET_AREA_TEXT, 
                (reset_left + 10, UIConfig.RESET_AREA_HEIGHT - 10), 
                UIConfig.COLOR_BLACK, 
                UIConfig.RESET_AREA_FONT_SIZE
            )
            
        # 绘制轨迹
        if viz_config.get('show_tracking_path', True) and len(position_history) >= 2:
            for i in range(1, len(position_history)):
                pt1 = (int(position_history[i-1][0] * frame.shape[1]), 
                      int(position_history[i-1][1] * frame.shape[0]))
                pt2 = (int(position_history[i][0] * frame.shape[1]), 
                      int(position_history[i][1] * frame.shape[0]))
                cv2.line(frame, pt1, pt2, UIConfig.TRACK_COLOR, UIConfig.TRACK_THICKNESS)
            
        return frame 