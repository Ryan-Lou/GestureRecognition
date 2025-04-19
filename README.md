# 基于MediaPipe的智能手势控制系统

## 项目介绍

这是一个基于MediaPipe的智能手势控制系统，可通过摄像头识别手势来触发键盘操作。目前实现了多种手势控制，包括握拳/张开触发空格键以及二指并拢左右滑动触发方向键。系统可用于视频播放控制等场景，具有实时性好、准确率高、可维护性强等特点。

## 主要功能

- 实时手部检测与关键点识别
- 多种手势状态判断：
  - 手掌张开/握拳
  - 二指并拢（食指与中指伸直并拢，其余手指收回）
  - 拇指向上/向下（拇指伸直指向上/下方，其余手指弯曲）
- 多种触发动作：
  - 从张开到握拳触发空格键（播放/暂停）
  - 二指并拢向右滑动触发右方向键（快进）
  - 二指并拢向左滑动触发左方向键（回退）
  - 拇指向上触发上方向键（增加音量）
  - 拇指向下触发下方向键（减少音量）
- 可视化界面（显示手部骨架、状态、开合度、滑动轨迹、音量条等信息）
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

3. 下载MediaPipe手势识别模型

```bash
# 创建模型目录
mkdir -p models

# 下载手势识别模型
# 可以从 MediaPipe 官方网站下载 gesture_recognizer.task 模型
# 下载地址: https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer/index#models
# 将下载的模型文件放在 models/ 目录下
```

4. 运行主程序

```bash
python src/main.py
```

## 使用方法

1. 运行主程序

```bash
python src/main.py
```

2. 手势操作说明
   
   **基础手势：手掌张开/握拳**
   - 将手掌放在摄像头视野内
   - 完全张开手掌（开合度>0.7）
   - 握拳（开合度<0.3）
   - 此时会触发空格键（播放/暂停）
   - 再次触发需要先张开手掌再握拳

   **二指敬礼手势（V手势）滑动**
   - 将食指和中指伸直并拢，其余手指收回
   - 向右滑动：触发右方向键（快进）
   - 向左滑动：触发左方向键（回退）
   - 滑动距离需超过屏幕宽度的10%才会触发
   
   **拇指手势音量控制**
   - 拇指向上，其余四指弯曲：触发上方向键（增加音量）
   - 拇指向下，其余四指弯曲：触发下方向键（减少音量）
   - 每次触发会增减5%音量（可在配置中调整）
   - 屏幕上会显示当前音量百分比

   **退出程序**
   - 按'q'键退出程序

## 配置文件说明

通过修改`config.yaml`文件可以调整系统参数：

```yaml
thresholds:
  fist_max: 0.3  # 握拳阈值上限
  open_min: 0.7  # 张开阈值下限
cooldown: 1.0    # 触发冷却时间（秒）
use_system_cmd: true  # 是否使用系统命令触发按键
salute_trigger_threshold: 0.1  # 二指敬礼滑动触发阈值（屏幕宽度的10%）

# MediaPipe手势识别设置
use_mediapipe_gesture: true  # 是否使用MediaPipe手势识别
mediapipe_gesture_model_path: 'models/gesture_recognizer.task'  # 手势识别模型路径
gesture_confidence_threshold: 0.7  # 手势识别置信度阈值

camera:
  id: 0          # 摄像头ID
  width: 640     # 画面宽度
  height: 480    # 画面高度
  fps: 30        # 帧率

visualization:
  show_landmarks: true  # 显示手部关键点
  show_state: true      # 显示状态信息
  show_progress: true   # 显示进度条
  show_volume_bar: true # 显示音量条
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
- MediaPipe Gesture Recognizer：用于精确识别手势，特别是拇指向上/向下手势
- 手势开合度算法：基于手指指尖到指根的距离与指根到手腕的距离的比值
- 二指敬礼手势识别：基于手指伸直程度和并拢程度的计算
- 滑动检测算法：基于手掌中心点的水平位移计算
- 状态防抖机制：需要连续3帧相同状态才确认
- 冷却时间机制：防止短时间内重复触发

## 最近更新

- 集成MediaPipe Gesture Recognizer，提高拇指手势识别准确率，减少误触发
- 实现了拇指向上/向下手势控制音量功能
- 实现了V形手势左右滑动触发左右方向键功能，可用于视频前进/后退控制
- 优化了手势识别算法，提高了V形手势和拇指手势识别的准确率
- 改进了手势滑动检测逻辑，减少误触发
- 完善了可视化界面，增加了手势轨迹显示和音量控制反馈

## 待实现功能

- 更多手势的支持（如旋转、上下滑动等）
- 自定义按键映射
- 更高级的手势识别算法
- GUI配置界面
- 多平台支持

## 许可证

[MIT License](LICENSE)

## 贡献

欢迎提交问题报告和功能请求！ 