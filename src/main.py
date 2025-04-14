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
from src.ui_utils import UIDrawer, UIConfig

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
        # 初始化相机
        if not self.camera.init_camera():
            print("相机初始化失败，请检查配置和连接")
            sys.exit(1)
            
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
            
            while self.running:
                # 读取图像帧
                ret, frame = self.camera.read_frame()
                if not ret or frame is None:
                    if console_config.get('show_error_messages', True):
                        print("无法读取摄像头画面")
                    break
                
                # 更新帧率
                self._update_fps()
                
                # 分析手势
                state, openness, is_transition, extra_gestures = self.analyzer.analyze_frame(frame)
                
                # 绘制手部关键点
                frame = UIDrawer.draw_hand_landmarks(
                    frame=frame,
                    landmarks_results=self.analyzer.last_results,
                    extra_info=extra_gestures,
                    mp_draw=self.analyzer.mp_draw,
                    mp_hands=self.analyzer.mp_hands,
                    center_position=self.analyzer.center_position,
                    reset_threshold=self.analyzer.reset_threshold,
                    direction_locked=self.analyzer.direction_locked,
                    swipe_complete=self.analyzer.swipe_complete,
                    position_history=self.analyzer.position_history,
                    viz_config=viz_config
                )
                
                # 绘制UI
                frame = UIDrawer.draw_ui_elements(
                    frame=frame,
                    state=state,
                    openness=openness,
                    fps=self.fps,
                    analyzer_instance=self.analyzer,
                    viz_config=viz_config
                )
                
                # 检查并触发动作 - 从张开到握拳触发空格键
                if is_transition:
                    triggered = self.trigger.trigger_space()
                    if console_config.get('show_trigger_events', True):
                        print("触发空格键!")
                    
                    # 在画面上显示触发提示
                    if viz_config.get('show_trigger_info', True):
                        frame = UIDrawer.draw_trigger_notification(
                            frame=frame,
                            text="空格键!",
                            y_offset=0,
                            color=UIConfig.TRIGGER_TEXT_COLOR
                        )
                
                # 检查V手势滑动手势
                if extra_gestures.get("vsign_swiped", False):
                    direction = extra_gestures.get("swipe_direction")
                    if direction:
                        self.trigger.trigger_direction_key(direction)
                        
                        # 在画面上显示触发提示
                        if viz_config.get('show_trigger_info', True):
                            display_text = f"{'右' if direction == 'right' else '左'}方向键!"
                            frame = UIDrawer.draw_trigger_notification(
                                frame=frame,
                                text=display_text,
                                y_offset=UIConfig.TRIGGER_Y_OFFSET,
                                color=UIConfig.TRIGGER_DIR_COLOR
                            )
                
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