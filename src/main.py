import cv2
import numpy as np
from typing import Optional
import sys
import os
import yaml

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
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _draw_ui(self, frame: np.ndarray, state: Optional[str], openness: float) -> np.ndarray:
        """
        绘制用户界面
        
        Args:
            frame: 输入图像帧
            state: 当前手势状态
            openness: 手掌开合度
            
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
        
        return frame
    
    def run(self):
        """运行手势控制系统"""
        try:
            while self.running:
                # 读取图像帧
                frame = self.camera.read_frame()
                if frame is None:
                    print("无法读取摄像头画面")
                    break
                
                # 分析手势
                state, openness = self.analyzer.analyze_frame(frame)
                
                # 检查并触发动作
                if state and self.analyzer.last_state:
                    self.trigger.check_and_trigger(state, self.analyzer.last_state)
                
                # 绘制UI
                frame = self._draw_ui(frame, state, openness)
                
                # 显示图像
                cv2.imshow(self.config['visualization']['window_name'], frame)
                
                # 检查退出条件
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.camera.release()
        cv2.destroyAllWindows()

def main():
    """主函数"""
    config_path = "config.yaml"
    controller = GestureController(config_path)
    controller.run()

if __name__ == "__main__":
    main() 