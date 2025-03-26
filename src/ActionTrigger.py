from pynput.keyboard import Controller, Key
import time
from typing import Optional
import yaml

class ActionTrigger:
    """动作触发类，负责键盘事件的控制"""
    
    def __init__(self, config_path: str):
        """
        初始化动作触发器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.keyboard = Controller()
        self.last_trigger_time = 0
        self.cooldown = self.config['cooldown']
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def check_and_trigger(self, current_state: Optional[str], last_state: Optional[str]) -> bool:
        """
        检查并触发动作
        
        Args:
            current_state: 当前手势状态
            last_state: 上一个手势状态
            
        Returns:
            bool: 是否触发了动作
        """
        # 检查状态转换
        if (last_state == 'fist' and current_state == 'open'):
            current_time = time.time()
            
            # 检查冷却时间
            if current_time - self.last_trigger_time >= self.cooldown:
                self.trigger_space()
                self.last_trigger_time = current_time
                return True
                
        return False
    
    def trigger_space(self) -> None:
        """触发空格键"""
        self.keyboard.press(Key.space)
        self.keyboard.release(Key.space) 