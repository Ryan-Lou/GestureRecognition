# 基于MediaPipe的手势控制系统

一个基于MediaPipe和OpenCV的实时手势控制系统，支持多种手势识别和交互控制。

## 功能特点

- **实时手势识别**：使用MediaPipe实现高精度手部关键点检测
- **多种手势支持**：
  - 手掌开合控制（空格键）
  - V手势滑动控制（左右方向键）
  - 拇指手势控制（上下方向键）
- **智能防抖动**：采用连续帧验证和冷却时间机制，避免误触发
- **实时可视化**：提供丰富的视觉反馈，包括手部骨架、状态显示等
- **高度可配置**：通过YAML配置文件灵活调整系统参数
- **多线程优化**：采用生产者-消费者模式，确保实时性能
- **跨平台支持**：支持Windows系统（键盘触发功能）

## 系统要求

- Python 3.7+
- Windows 系统（键盘触发功能）
- 摄像头

## 安装步骤

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
   - 完全张开手掌（开合度>0.6）
   - 握拳（开合度<0.2）
   - 此时会触发空格键（播放/暂停）
   - 再次触发需要先张开手掌再握拳

   **V手势滑动**
   - 将食指和中指伸直并拢（V手势），其余手指收回
   - 向右滑动：触发右方向键（前进）
   - 向左滑动：触发左方向键（后退）
   - 滑动距离需超过屏幕宽度的20%才会触发
   - 滑动后需要将手移回中心区域才能再次触发
   
   **拇指手势控制**
   - 拇指向上，其余四指弯曲：触发上方向键
   - 拇指向下，其余四指弯曲：触发下方向键
   - 需要连续5帧检测到同样的拇指手势才会触发
   - 触发后需要等待30帧的冷却时间才能再次触发
   - 使用拇指角度验证（向上<45度，向下>135度）防止误触

   **退出程序**
   - 按'q'键退出程序

## 项目结构

```
├── config.yaml         # 配置文件
├── requirements.txt    # 依赖文件
├── README.md          # 项目说明文档
└── src/               # 源代码目录
    ├── main.py        # 主程序入口
    ├── CameraManager.py    # 相机管理模块
    ├── GestureAnalyzer.py  # 手势分析模块
    ├── ActionTrigger.py    # 动作触发模块
    └── ui_utils.py    # UI工具模块
```

## 配置说明

通过修改`config.yaml`文件可以调整系统参数：

```yaml
thresholds:
  fist_max: 0.2  # 握拳阈值上限
  open_min: 0.6  # 张开阈值下限

# 触发控制参数
cooldown: 1.0  # 冷却时间（秒）
vsign_trigger_threshold: 0.2  # V手势滑动触发阈值（屏幕宽度的20%）
thumb_consecutive_required: 5  # 拇指手势需要连续几帧才能触发
thumb_gesture_cooldown: 30  # 拇指手势冷却帧数

# 可视化配置
visualization:
  show_landmarks: true  # 显示手部关键点
  show_state: true      # 显示状态信息
  show_trigger_history: true  # 显示触发历史
  show_trigger_notification: true  # 显示触发提示
```

## 常见问题

1. **Q: 手势识别不准确？**
   - A: 确保光线充足，手部在摄像头视野内清晰可见
   - A: 调整`thresholds`参数以适应您的使用环境

2. **Q: 拇指手势与握拳容易混淆？**
   - A: 调整`thumb_gesture.fist_distinction_factor`参数，增大该值
   - A: 确保`thumb_gesture.straightness_threshold`设置适当（建议0.80）

3. **Q: V手势滑动不触发或过于敏感？**
   - A: 调整`vsign_trigger_threshold`参数，默认为0.2（屏幕宽度的20%）

4. **Q: 无法触发键盘事件？**
   - A: 确保程序有足够权限
   - A: 尝试设置`use_system_cmd: true`

5. **Q: 拇指手势触发太频繁？**
   - A: 增加`thumb_gesture_cooldown`（目前为30帧）
   - A: 增加`thumb_consecutive_required`（目前为5帧）

## 技术特点

1. **高度参数化配置**
   - 使用YAML配置文件管理所有参数
   - 支持运行时动态调整

2. **多线程优化**
   - 采用生产者-消费者模式
   - 队列长度限制防止内存溢出
   - 非阻塞式队列操作避免线程卡死

3. **智能防抖动机制**
   - 连续帧验证
   - 冷却时间控制
   - 状态锁定机制

4. **实时可视化反馈**
   - 手部骨架绘制
   - 状态信息显示
   - 触发历史记录
   - 实时FPS显示

## 贡献指南

欢迎提交问题报告和功能请求！如果您想为项目做出贡献，请遵循以下步骤：

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

[MIT License](LICENSE)

## 致谢

- [MediaPipe](https://mediapipe.dev/) - 提供手部关键点检测
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [pynput](https://github.com/moses-palmer/pynput) - 键盘控制库 