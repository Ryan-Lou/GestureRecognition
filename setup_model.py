#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MediaPipe手势识别模型下载脚本

此脚本用于下载MediaPipe手势识别模型并将其保存到项目目录中。
它会检查模型文件是否已存在，如果不存在则进行下载。
使用Python标准库urllib替代requests，避免额外依赖。
"""

import os
import sys
import urllib.request
from pathlib import Path

# MediaPipe手势识别模型下载URL
GESTURE_RECOGNIZER_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"

# 模型保存路径
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_recognizer.task")

def download_file(url, dest_path):
    """下载文件并显示进度"""
    print(f"正在从 {url} 下载模型...")
    
    # 创建进度显示回调函数
    def show_progress(count, block_size, total_size):
        downloaded = count * block_size
        # 避免除以零错误
        if total_size > 0:
            percent = min(100, int(downloaded * 100 / total_size))
            progress = int(50 * percent / 100)
            progress_bar = '█' * progress + '░' * (50 - progress)
            sys.stdout.write(f"\r|{progress_bar}| {percent}% ({downloaded}/{total_size} bytes)")
            sys.stdout.flush()
    
    # 下载文件
    try:
        urllib.request.urlretrieve(url, dest_path, show_progress)
        print("\n下载完成!")
    except Exception as e:
        print(f"\n下载失败: {e}")
        raise

def main():
    """主函数，检查并下载模型"""
    # 确保模型目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 检查模型文件是否已存在
    if os.path.exists(MODEL_PATH):
        print(f"模型文件已存在: {MODEL_PATH}")
        user_input = input("是否重新下载? (y/n): ")
        if user_input.lower() != 'y':
            print("跳过下载.")
            return
    
    try:
        # 下载模型文件
        download_file(GESTURE_RECOGNIZER_URL, MODEL_PATH)
        print(f"模型已保存到: {MODEL_PATH}")
    except Exception as e:
        print(f"下载模型时出错: {e}")
        sys.exit(1)
    
    print("\n设置完成! 现在您可以运行程序使用MediaPipe手势识别功能了.")
    print("运行命令: python src/main.py")

if __name__ == "__main__":
    main() 