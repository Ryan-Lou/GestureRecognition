import cv2
import numpy as np
from typing import Optional
import sys
import os
import yaml
import time
import win32gui
import win32con

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
        
        # 绘制状态文本
        if state:
            color = (0, 255, 0) if state == 'open' else (0, 0, 255)
            cv2.putText(frame, f"State: {state}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # 绘制开合度进度条
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = height - 40
        
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
        openness_text = f"Openness: {openness:.2f}"
        cv2.putText(frame, openness_text, (bar_x + bar_width + 10, bar_y + bar_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # 绘制FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (width - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示最新状态
        last_state = self.analyzer.last_state
        current_state = self.analyzer.current_state
        if last_state and current_state:
            state_text = f"Last: {last_state}, Current: {current_state}"
            cv2.putText(frame, state_text, (10, height - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        # 显示手势说明
        instructions = [
            "Gestures:",
            "1. Open -> Fist: Space",
            "2. V-Sign Right: Forward",
            "3. V-Sign Left: Backward"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (width - 250, 90 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
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
            print("按Q键退出程序")
            print("等待检测到手势...")
            print("支持的手势：")
            print("1. 从张开到握拳：触发空格键")
            print("2. V手势向右滑动：触发右方向键")
            print("3. V手势向左滑动：触发左方向键")
            
            while self.running:
                # 读取图像帧
                frame = self.camera.read_frame()
                if frame is None:
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
                    print("触发空格键!")
                    # 在画面上显示触发提示
                    cv2.putText(frame, "SPACE!", (frame.shape[1]//2 - 100, frame.shape[0]//2),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                # 检查V手势滑动手势
                if extra_gestures.get("vsign_swiped", False):
                    direction = extra_gestures.get("swipe_direction")
                    if direction:
                        self.trigger.trigger_direction_key(direction)
                        # 在画面上显示触发提示
                        display_text = f"{direction.upper()}!"
                        cv2.putText(frame, display_text, (frame.shape[1]//2 - 100, frame.shape[0]//2 + 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                
                # 绘制UI
                frame = self._draw_ui(frame, state, openness, extra_gestures)
                
                # 显示图像
                cv2.imshow(self.window_name, frame)
                
                # 检查退出条件
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.camera.release()
        cv2.destroyAllWindows()
        print("程序已退出")

def main():
    """主函数"""
    config_path = "config.yaml"
    controller = GestureController(config_path)
    controller.run()

if __name__ == "__main__":
    main()