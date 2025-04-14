from pynput.keyboard import Controller, Key
import time
from typing import Optional
import yaml
import os

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
    
    def trigger_space(self) -> bool:
        """
        触发空格键
        
        Returns:
            bool: 是否成功触发
        """
        current_time = time.time()
        
        # 检查冷却时间
        if current_time - self.last_trigger_time < self.cooldown:
            return False
            
        try:
            # 方法1: 使用pynput
            self.keyboard.press(Key.space)
            time.sleep(0.1)  # 增加延迟
            self.keyboard.release(Key.space)
            
            # 方法2: 使用系统命令（备用方案）
            if self.config.get('use_system_cmd', False):
                os.system('powershell -command "$wshell = New-Object -ComObject wscript.shell; $wshell.SendKeys(\' \')"')
                
            self.last_trigger_time = current_time
            return True
        except Exception as e:
            # 获取终端输出配置
            console_config = self.config.get('console_output', {})
            if console_config.get('show_error_messages', True):
                print(f"触发空格键出错: {e}")
            return False
            
    def trigger_direction_key(self, direction: str) -> bool:
        """
        触发方向键
        
        Args:
            direction: 方向，"left"或"right"
            
        Returns:
            bool: 是否成功触发
        """
        current_time = time.time()
        
        # 检查冷却时间
        if current_time - self.last_trigger_time < self.cooldown:
            return False
            
        try:
            # 根据方向选择按键
            key = Key.left if direction == "left" else Key.right
            
            # 方法1: 使用pynput
            self.keyboard.press(key)
            time.sleep(0.1)  # 增加延迟
            self.keyboard.release(key)
            
            # 方法2: 使用系统命令（备用方案）
            if self.config.get('use_system_cmd', False):
                key_code = "{LEFT}" if direction == "left" else "{RIGHT}"
                os.system(f'powershell -command "$wshell = New-Object -ComObject wscript.shell; $wshell.SendKeys(\'{key_code}\')"')
                
            # 更新最后触发时间
            self.last_trigger_time = current_time
            
            # 获取终端输出配置
            console_config = self.config.get('console_output', {})
            if console_config.get('show_trigger_events', True):
                print(f"触发{direction}方向键!")
                
            return True
        except Exception as e:
            # 获取终端输出配置
            console_config = self.config.get('console_output', {})
            if console_config.get('show_error_messages', True):
                print(f"触发方向键出错: {e}")
            return False 