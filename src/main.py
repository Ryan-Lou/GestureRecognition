import cv2
import numpy as np
from typing import Optional
import sys
import os
import yaml
import time
import win32gui
import win32con
from PIL import Image, ImageDraw, ImageFont

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.CameraManager import CameraManager
from src.GestureAnalyzer import GestureAnalyzer
from src.ActionTrigger import ActionTrigger

class GestureController:
    """手势控制器主类"""
    
    def __init__(self, config_path: str):
        """
        初始化手势控制器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.camera = CameraManager(config_path)
        self.analyzer = GestureAnalyzer(config_path)
        self.trigger = ActionTrigger(config_path)
        self.running = True
        
        # 帧率计算相关变量
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.fps = 0
        self.fps_update_interval = 1  # 每1秒更新一次FPS
        self.last_fps_update = time.time()
        self.frame_count = 0
        
        # 创建窗口
        self.window_name = self.config['visualization']['window_name']
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # 设置窗口置顶
        hwnd = win32gui.FindWindow(None, self.window_name)
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
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
    
    def _draw_ui(self, frame: np.ndarray, state: Optional[str], openness: float, extra_info=None) -> np.ndarray:
        """
        绘制用户界面
        
        Args:
            frame: 输入图像帧
            state: 当前手势状态
            openness: 手掌开合度
            extra_info: 额外手势信息
            
        Returns:
            np.ndarray: 绘制了UI的图像帧
        """
        height, width = frame.shape[:2]
        
        # 获取可视化配置
        viz_config = self.config.get('visualization', {})
        
        # UI元素位置和样式参数（可在此处调整）
        # 状态文本配置
        state_position = [10, 30]         # 状态文本位置 [x, y]
        state_font_size = 30              # 状态文本字体大小
        
        # 进度条配置
        bar_width = 200                   # 开合度进度条宽度
        bar_height = 20                   # 开合度进度条高度
        bar_x_offset = 10                 # 开合度进度条X坐标偏移
        bar_y_offset = 40                 # 开合度进度条Y坐标偏移（从底部开始算）
        
        # FPS显示配置
        fps_position = [150, 30]          # FPS显示位置 [x偏移量, y]
        fps_font_size = 30                # FPS字体大小
        
        # 历史状态配置
        history_position = [10, 70]       # 历史状态位置 [x, y偏移量（从底部开始算）]
        history_font_size = 20            # 历史状态字体大小
        
        # 手势说明配置
        instr_x_offset = 250              # 手势说明X偏移量（从右边开始算）
        instr_y_start = 90                # 手势说明起始Y坐标
        instr_line_spacing = 30           # 手势说明行间距
        instr_font_size = 20              # 手势说明字体大小
        
        # 触发提示配置
        trigger_font_size = 45            # 触发提示字体大小
        trigger_y_offset = 50             # 第二行触发提示的Y偏移量
        
        # 绘制状态文本
        if state and viz_config.get('show_state', True):
            color = (0, 255, 0) if state == 'open' else (0, 0, 255)
            state_text = f"状态: {'张开' if state == 'open' else '握拳'}"
            frame = self._cv2_put_chinese_text(
                frame, 
                state_text, 
                (state_position[0], state_position[1]), 
                color, 
                state_font_size
            )
        
        # 绘制开合度进度条
        if viz_config.get('show_openness_bar', True):
            # 计算进度条位置
            bar_x = bar_x_offset
            bar_y = height - bar_y_offset
            
            # 背景
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + bar_width, bar_y + bar_height),
                         (255, 255, 255), -1)
            
            # 进度
            progress_width = int(bar_width * openness)
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + progress_width, bar_y + bar_height),
                         (0, 255, 0), -1)
            
            # 边框
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + bar_width, bar_y + bar_height),
                         (0, 0, 0), 2)
            
            # 显示开合度数值
            openness_text = f"开合度: {openness:.2f}"
            frame = self._cv2_put_chinese_text(
                frame, 
                openness_text, 
                (bar_x + bar_width + 10, bar_y + bar_height), 
                (0, 0, 0), 
                20
            )
        
        # 绘制FPS
        if viz_config.get('show_fps', True):
            fps_text = f"FPS: {self.fps:.1f}"
            frame = self._cv2_put_chinese_text(
                frame, 
                fps_text, 
                (width - fps_position[0], fps_position[1]), 
                (0, 255, 0), 
                fps_font_size
            )
        
        # 显示最新状态
        if viz_config.get('show_gesture_history', True):
            last_state = self.analyzer.last_state
            current_state = self.analyzer.current_state
            if last_state and current_state:
                last_state_text = '张开' if last_state == 'open' else '握拳'
                current_state_text = '张开' if current_state == 'open' else '握拳'
                state_text = f"上次: {last_state_text}, 当前: {current_state_text}"
                frame = self._cv2_put_chinese_text(
                    frame, 
                    state_text, 
                    (history_position[0], height - history_position[1]), 
                    (0, 0, 255), 
                    history_font_size
                )
            
        # 显示手势说明
        if viz_config.get('show_instructions', True):
            instructions = [
                "手势说明:",
                "1. 从张开到握拳: 空格键",
                "2. V手势向右滑动: 前进",
                "3. V手势向左滑动: 后退"
            ]
            
            for i, text in enumerate(instructions):
                pos_y = instr_y_start + i * instr_line_spacing
                frame = self._cv2_put_chinese_text(
                    frame, 
                    text, 
                    (width - instr_x_offset, pos_y), 
                    (0, 0, 0), 
                    instr_font_size
                )
        
        return frame
    
    def _update_fps(self) -> None:
        """更新帧率计算"""
        current_time = time.time()
        self.frame_count += 1
        
        if current_time - self.last_fps_update >= self.fps_update_interval:
            self.fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time
    
    def run(self):
        """运行手势控制系统"""
        try:
            # 获取控制台输出配置
            console_config = self.config.get('console_output', {})
            
            # 显示启动信息
            if console_config.get('show_startup_info', True):
                print("按Q键退出程序")
                print("等待检测到手势...")
                print("支持的手势：")
                print("1. 从张开到握拳：触发空格键")
                print("2. V手势向右滑动：触发右方向键")
                print("3. V手势向左滑动：触发左方向键")
            
            # 获取可视化配置
            viz_config = self.config.get('visualization', {})
            
            # 触发提示配置
            trigger_font_size = 45            # 触发提示字体大小
            trigger_y_offset = 50             # 第二行触发提示的Y偏移量
            
            while self.running:
                # 读取图像帧
                frame = self.camera.read_frame()
                if frame is None:
                    if console_config.get('show_error_messages', True):
                        print("无法读取摄像头画面")
                    break
                
                # 更新帧率
                self._update_fps()
                
                # 分析手势
                state, openness, is_transition, extra_gestures = self.analyzer.analyze_frame(frame)
                
                # 绘制手部关键点
                frame = self.analyzer.draw_landmarks(frame, extra_gestures)
                
                # 检查并触发动作 - 从张开到握拳触发空格键
                if is_transition:
                    triggered = self.trigger.trigger_space()
                    if console_config.get('show_trigger_events', True):
                        print("触发空格键!")
                    
                    # 在画面上显示触发提示
                    if viz_config.get('show_trigger_info', True):
                        frame = self._cv2_put_chinese_text(
                            frame, 
                            "空格键!", 
                            (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                            (0, 0, 255), 
                            trigger_font_size
                        )
                
                # 检查V手势滑动手势
                if extra_gestures.get("vsign_swiped", False):
                    direction = extra_gestures.get("swipe_direction")
                    if direction:
                        self.trigger.trigger_direction_key(direction)
                        
                        # 在画面上显示触发提示
                        if viz_config.get('show_trigger_info', True):
                            display_text = f"{'右' if direction == 'right' else '左'}方向键!"
                            frame = self._cv2_put_chinese_text(
                                frame, 
                                display_text, 
                                (frame.shape[1]//2 - 100, frame.shape[0]//2 + trigger_y_offset), 
                                (255, 0, 0), 
                                trigger_font_size
                            )
                
                # 绘制UI
                frame = self._draw_ui(frame, state, openness, extra_gestures)
                
                # 显示图像
                cv2.imshow(self.window_name, frame)
                
                # 检查退出条件
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            if console_config.get('show_error_messages', True):
                print("\n程序被用户中断")
        except Exception as e:
            if console_config.get('show_error_messages', True):
                print(f"发生错误: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.camera.release()
        cv2.destroyAllWindows()
        
        # 获取控制台输出配置
        console_config = self.config.get('console_output', {})
        if console_config.get('show_error_messages', True):
            print("程序已退出")

def main():
    """主函数"""
    config_path = "config.yaml"
    controller = GestureController(config_path)
    controller.run()

if __name__ == "__main__":
    main()