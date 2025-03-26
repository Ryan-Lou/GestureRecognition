# 基于MediaPipe的智能手势控制系统

## 项目介绍

这是一个基于MediaPipe的智能手势控制系统，可通过摄像头识别手势来触发键盘操作。目前实现了通过手势（从张开到握拳）触发空格键，可用于控制媒体播放/暂停等功能。系统具有实时性好、准确率高、可维护性强等特点。

## 主要功能

- 实时手部检测与关键点识别
- 手势状态判断（张开/握拳）
- 状态转换触发（从张开到握拳触发空格键）
- 可视化界面（显示手部骨架、状态、开合度等信息）
- 跨应用键盘事件触发

## 系统要求

- Python 3.7+
- Windows 系统（对于键盘触发功能）
- 摄像头

## 安装

1. 克隆仓库

```bash
git clone https://github.com/yourusername/gesture-control.git
cd gesture-control
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 运行主程序

```bash
python src/main.py
```

2. 操作说明
   - 将手掌放在摄像头视野内
   - 完全张开手掌（开合度>0.7）
   - 握拳（开合度<0.3）
   - 此时会触发空格键
   - 再次触发需要先张开手掌再握拳
   - 按'q'键退出程序

## 配置文件说明

通过修改`config.yaml`文件可以调整系统参数：

```yaml
thresholds:
  fist_max: 0.3  # 握拳阈值上限
  open_min: 0.7  # 张开阈值下限
cooldown: 1.0    # 触发冷却时间（秒）
use_system_cmd: true  # 是否使用系统命令触发按键
camera:
  id: 0          # 摄像头ID
  width: 640     # 画面宽度
  height: 480    # 画面高度
  fps: 30        # 帧率
visualization:
  show_landmarks: true  # 显示手部关键点
  show_state: true      # 显示状态信息
  show_progress: true   # 显示进度条
  window_name: "Gesture Control"  # 窗口名称
```

## 项目结构

```
├── config.yaml         # 配置文件
├── requirements.txt    # 依赖文件
└── src/
    ├── CameraManager.py    # 摄像头管理模块
    ├── GestureAnalyzer.py  # 手势分析模块
    ├── ActionTrigger.py    # 动作触发模块
    └── main.py             # 主程序
```

## 核心技术

- MediaPipe Hands：用于实时手部检测和关键点识别
- 手势开合度算法：基于手指指尖到指根的距离与指根到手腕距离的比值
- 状态防抖机制：需要连续3帧相同状态才确认
- 冷却时间机制：防止短时间内重复触发

## 待实现功能

- 更多手势的支持
- 自定义按键映射
- 更高级的手势识别算法
- GUI配置界面
- 多平台支持

## 许可证

[MIT License](LICENSE)

## 贡献

欢迎提交问题报告和功能请求！ 