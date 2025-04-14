import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
import sys
import os
import yaml
import time
import win32gui
import win32con
import threading
import queue
from collections import deque

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
        
        # 多线程处理相关
        self.frame_queue = queue.Queue(maxsize=2)  # 原始帧队列
        self.result_queue = queue.Queue(maxsize=5)  # 处理结果队列
        self.processed_frames = deque(maxlen=2)  # 处理后的帧缓存，限制长度避免内存泄漏
        self.lock = threading.Lock()  # 线程锁
        
        # 线程是否活跃的标志
        self.is_processing_active = False
        
        # 控制台和可视化配置
        self.console_config = self.config.get('console_output', {})
        self.viz_config = self.config.get('visualization', {})
        
        # 缓存当前状态，避免重复渲染
        self.cached_state = None
        self.cached_openness = 0.0
        
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
    
    def _process_frames(self):
        """
        处理图像帧的线程函数
        从队列中获取帧，进行手势分析，将结果放入结果队列
        """
        while self.is_processing_active:
            try:
                if not self.frame_queue.empty():
                    # 获取一帧图像
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    # 分析手势
                    state, openness, is_transition, extra_gestures = self.analyzer.analyze_frame(frame)
                    
                    # 将原始帧和处理结果一起放入结果队列
                    result = {
                        "frame": frame,
                        "state": state,
                        "openness": openness,
                        "is_transition": is_transition,
                        "extra_gestures": extra_gestures,
                        "timestamp": time.time()
                    }
                    
                    # 使用非阻塞方式放入队列，避免处理线程卡住
                    try:
                        self.result_queue.put(result, block=False)
                    except queue.Full:
                        # 如果队列已满，丢弃最旧的结果
                        try:
                            self.result_queue.get(block=False)
                            self.result_queue.put(result, block=False)
                        except (queue.Empty, queue.Full):
                            pass
                    
                    # 标记这个任务已完成
                    self.frame_queue.task_done()
                else:
                    # 如果队列为空，短暂休眠避免CPU空转
                    time.sleep(0.001)
            except queue.Empty:
                # 超时后继续循环
                continue
            except Exception as e:
                if self.console_config.get('show_error_messages', True):
                    print(f"处理帧异常: {e}")
    
    def _render_ui(self, frame, state, openness, is_transition, extra_gestures):
        """
        渲染UI界面
        
        将所有UI绘制操作集中在一个方法中，便于管理
        
        Args:
            frame: 原始图像帧
            state: 手势状态
            openness: 开合度
            is_transition: 是否是状态转换
            extra_gestures: 额外手势信息
            
        Returns:
            np.ndarray: 绘制了UI的图像帧
        """
        # 创建帧的副本用于绘制，避免修改原始帧
        ui_frame = frame.copy()
        
        # 绘制手部关键点
        ui_frame = UIDrawer.draw_hand_landmarks(
            frame=ui_frame,
            landmarks_results=self.analyzer.last_results,
            extra_info=extra_gestures,
            mp_draw=self.analyzer.mp_draw,
            mp_hands=self.analyzer.mp_hands,
            center_position=self.analyzer.center_position,
            reset_threshold=self.analyzer.reset_threshold,
            direction_locked=self.analyzer.direction_locked,
            swipe_complete=self.analyzer.swipe_complete,
            position_history=self.analyzer.position_history,
            viz_config=self.viz_config
        )
        
        # 绘制UI
        ui_frame = UIDrawer.draw_ui_elements(
            frame=ui_frame,
            state=state,
            openness=openness,
            fps=self.fps,
            analyzer_instance=self.analyzer,
            viz_config=self.viz_config
        )
        
        # 检查并触发动作 - 从张开到握拳触发空格键
        if is_transition:
            triggered = self.trigger.trigger_space()
            if self.console_config.get('show_trigger_events', True):
                print("触发空格键!")
            
            # 在画面上显示触发提示
            if self.viz_config.get('show_trigger_info', True):
                ui_frame = UIDrawer.draw_trigger_notification(
                    frame=ui_frame,
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
                if self.viz_config.get('show_trigger_info', True):
                    display_text = f"{'右' if direction == 'right' else '左'}方向键!"
                    ui_frame = UIDrawer.draw_trigger_notification(
                        frame=ui_frame,
                        text=display_text,
                        y_offset=UIConfig.TRIGGER_Y_OFFSET,
                        color=UIConfig.TRIGGER_DIR_COLOR
                    )
                    
        return ui_frame
    
    def run(self):
        """运行手势控制系统"""
        try:
            # 显示启动信息
            if self.console_config.get('show_startup_info', True):
                print("按Q键退出程序")
                print("等待检测到手势...")
                print("支持的手势：")
                print("1. 从张开到握拳：触发空格键")
                print("2. V手势向右滑动：触发右方向键")
                print("3. V手势向左滑动：触发左方向键")
            
            # 启动处理线程
            self.is_processing_active = True
            processing_thread = threading.Thread(target=self._process_frames, daemon=True)
            processing_thread.start()
            
            # 主循环 - 处理图像显示和用户输入
            while self.running:
                # 读取图像帧
                ret, frame = self.camera.read_frame()
                if not ret or frame is None:
                    if self.console_config.get('show_error_messages', True):
                        print("无法读取摄像头画面")
                    break
                
                # 将帧放入处理队列
                try:
                    self.frame_queue.put(frame.copy(), block=False)
                except queue.Full:
                    # 如果队列已满，丢弃一些旧帧
                    try:
                        self.frame_queue.get(block=False)
                        self.frame_queue.put(frame.copy(), block=False)
                    except (queue.Empty, queue.Full):
                        pass
                
                # 更新帧率
                self._update_fps()
                
                # 从结果队列获取处理结果
                result = None
                try:
                    # 非阻塞方式获取结果，避免主线程卡住
                    result = self.result_queue.get(block=False)
                    self.result_queue.task_done()
                except queue.Empty:
                    # 如果没有新结果，使用最后一个处理过的帧
                    if self.processed_frames:
                        last_frame = self.processed_frames[-1]
                        cv2.imshow(self.window_name, last_frame)
                        
                        # 检查退出条件
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                            
                        # 短暂休眠避免CPU占用过高
                        time.sleep(0.001)
                        continue
                
                # 如果有新结果，渲染UI
                if result:
                    result_frame = result["frame"]
                    state = result["state"]
                    openness = result["openness"]
                    is_transition = result["is_transition"]
                    extra_gestures = result["extra_gestures"]
                    
                    # 渲染UI
                    ui_frame = self._render_ui(result_frame, state, openness, is_transition, extra_gestures)
                    
                    # 存储处理后的帧
                    self.processed_frames.append(ui_frame)
                    
                    # 显示图像
                    cv2.imshow(self.window_name, ui_frame)
                
                # 检查退出条件
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            if self.console_config.get('show_error_messages', True):
                print("\n程序被用户中断")
        except Exception as e:
            if self.console_config.get('show_error_messages', True):
                print(f"发生错误: {e}")
        finally:
            # 停止处理线程
            self.is_processing_active = False
            
            # 等待处理线程结束
            if processing_thread.is_alive():
                processing_thread.join(timeout=1.0)
                
            # 清理资源
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        # 停止所有线程
        self.running = False
        self.is_processing_active = False
        
        # 释放相机资源
        self.camera.release()
        
        # 关闭所有窗口
        cv2.destroyAllWindows()
        
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.task_done()
            except queue.Empty:
                break
                
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
                self.result_queue.task_done()
            except queue.Empty:
                break
        
        # 获取控制台输出配置
        if self.console_config.get('show_error_messages', True):
            print("程序已退出")

def main():
    """主函数"""
    config_path = "config.yaml"
    controller = GestureController(config_path)
    controller.run()

if __name__ == "__main__":
    main()