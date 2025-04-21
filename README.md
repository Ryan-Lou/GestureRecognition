# 基于MediaPipe的智能手势控制系统

## 项目介绍

这是一个基于MediaPipe的智能手势控制系统，可通过摄像头识别手势来触发键盘操作。目前实现了多种手势控制，包括握拳/张开触发空格键、二指并拢左右滑动触发方向键以及拇指上下手势控制音量。系统可用于视频播放控制、演示控制等场景，具有实时性好、准确率高、可维护性强等特点。

## 主要功能

- **实时手部检测与关键点识别**：基于MediaPipe Hands模型，支持实时识别手部21个关键点
- **多种手势状态判断**：
  - 手掌张开/握拳：基于四指伸直程度的判断
  - V手势：食指与中指伸直并拢，其余手指收回
  - 拇指向上/向下：拇指伸直指向上方/下方，其余手指弯曲
- **多种触发动作**：
  - 从张开到握拳触发空格键（播放/暂停）
  - V手势向右滑动触发右方向键（前进）
  - V手势向左滑动触发左方向键（后退）
  - 拇指向上触发上方向键（增加音量）
  - 拇指向下触发下方向键（减少音量）
- **可视化界面**：
  - 显示手部骨架和关键点
  - 状态指示（当前手势状态及历史）
  - 手势开合度进度条
  - 音量控制反馈
  - 轨迹显示和动作提示
- **跨应用键盘事件触发**：可以控制任何支持键盘快捷键的应用程序

## 系统架构

项目采用模块化设计，主要组件包括：

- **CameraManager**：负责摄像头初始化和图像捕获
- **GestureAnalyzer**：手势分析核心，识别各种手势和状态
- **ActionTrigger**：负责触发键盘事件
- **UIDrawer**：处理所有可视化绘制工作
- **GestureController**：主控制器，协调各模块工作

采用多线程架构，将图像采集和手势处理分离，提高性能和响应速度。

## 系统要求

- Python 3.7+
- Windows 系统（键盘触发功能）
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

2. 手势操作说明
   
   **基础手势：手掌张开/握拳**
   - 将手掌放在摄像头视野内
   - 完全张开手掌（开合度>0.7）
   - 握拳（开合度<0.3）
   - 此时会触发空格键（播放/暂停）
   - 再次触发需要先张开手掌再握拳

   **V手势滑动**
   - 将食指和中指伸直并拢，其余手指收回
   - 向右滑动：触发右方向键（前进）
   - 向左滑动：触发左方向键（后退）
   - 滑动距离需超过屏幕宽度的10%才会触发
   - 滑动后需要将手移回中心区域才能再次触发
   
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
use_system_cmd: false  # 是否使用系统命令触发按键
vsign_trigger_threshold: 0.1  # V手势滑动触发阈值（屏幕宽度的10%）
center_position: 0.5    # 屏幕中心位置（归一化坐标）
reset_threshold: 0.1    # 复位阈值，接近中心位置的范围
thumb_gesture_cooldown: 15  # 拇指手势冷却帧数
volume_change_step: 5   # 每次音量变化步长（%）

# 拇指手势识别参数
thumb_gesture:
  straightness_threshold: 0.7  # 拇指伸直度阈值（0.0-1.0）
  bent_fingers_count: 3        # 判定为握拳状态需要弯曲的手指数量
  y_distance_threshold: 0.05   # 拇指尖与掌指关节的Y坐标差异阈值
  fist_distinction_factor: 0.3 # 拇指手势与握拳区分系数（越大区分越明显）

camera:
  camera_index: 0       # 摄像头ID
  width: 640            # 画面宽度
  height: 480           # 画面高度
  fps: 30               # 帧率
  flip_horizontal: true # 是否水平翻转画面

visualization:
  show_landmarks: true  # 显示手部关键点
  show_state: true      # 显示状态信息
  show_openness_bar: true # 显示进度条
  show_volume_bar: true # 显示音量条
  window_name: "Gesture Control"  # 窗口名称
```

## 项目结构

```
├── config.yaml         # 配置文件
├── requirements.txt    # 依赖文件
└── src/
    ├── main.py             # 主程序
    ├── CameraManager.py    # 摄像头管理模块
    ├── GestureAnalyzer.py  # 手势分析模块
    ├── ActionTrigger.py    # 动作触发模块
    ├── ui_utils.py         # UI绘制工具模块
    └── gesture_recognition.py  # 手势识别辅助函数
```

## 核心技术

- **MediaPipe Hands**：用于实时手部检测和关键点识别
- **手势开合度算法**：基于手指指尖到指根的距离与指根到手腕距离的比值计算
- **V手势识别**：基于手指伸直程度和并拢程度的计算
- **拇指手势识别**：基于拇指方向和其他手指弯曲程度的组合判断
- **滑动检测算法**：基于手部关键点的水平位移计算
- **多线程处理**：采用生产者-消费者模式，分离图像采集和处理
- **状态防抖机制**：需要连续多帧相同状态才确认，避免抖动
- **冷却时间机制**：防止短时间内重复触发

## 拇指手势调整指南

如果在使用过程中发现拇指手势与握拳之间的区分不够明显，可以调整以下参数：

1. **`thumb_gesture.fist_distinction_factor`**：
   - 增大此值可以要求拇指更明显地突出，减少误识别
   - 建议范围：0.3-0.6

2. **`thumb_gesture.y_distance_threshold`**：
   - 控制拇指需要多明显地向上/向下才被识别
   - 建议范围：0.05-0.1

3. **调试模式**：
   - 设置`console_output.show_thumb_debug: true`查看详细参数
   - 观察实际数据，针对性调整上述参数

## 常见问题

1. **Q: 手势识别不准确？**
   - A: 尝试在光线充足的环境下使用，确保手部在摄像头视野中清晰可见。

2. **Q: 拇指手势与握拳容易混淆？**
   - A: 调整`thumb_gesture.fist_distinction_factor`参数，增大该值。

3. **Q: 触发太灵敏或不够灵敏？**
   - A: 调整`cooldown`参数，值越小越灵敏。

4. **Q: 无法触发键盘事件？**
   - A: 确保程序有足够权限，或尝试设置`use_system_cmd: true`。

## 最近更新

- 实现了拇指向上/向下手势控制音量功能
- 优化了手势识别算法，提高了V形手势和拇指手势识别的准确率
- 改进了手势滑动检测逻辑，减少误触发
- 完善了可视化界面，增加了手势轨迹显示和音量控制反馈
- 增加了多层次的配置选项，适应不同使用场景

## 待实现功能

- 更多手势的支持（如旋转、捏合等）
- 自定义按键映射
- 更高级的手势识别算法
- GUI配置界面
- 多平台支持

## 贡献

欢迎提交问题报告和功能请求！如果您想为项目做出贡献，请遵循以下步骤：

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

[MIT License](LICENSE) 