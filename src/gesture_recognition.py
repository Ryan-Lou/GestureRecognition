import cv2
import mediapipe as mp
import time  # time模块用于计算帧率
# 初始化 MediaPipe Hands 模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # 动态模式，适合实时视频流
    max_num_hands=2,  # 支持同时检测两只手
    min_detection_confidence=0.7,  # 最小检测置信度
    min_tracking_confidence=0.5   # 最小跟踪置信度
)

mp_draw = mp.solutions.drawing_utils  # 用于绘制关键点和骨架

# 启动摄像头捕获
cap = cv2.VideoCapture(0)

# 记录前一帧的时间
prev_time = time.time()

while True:
    success, frame = cap.read()
    if not success:
        break

    # 翻转图像，确保手势方向与用户一致
    frame = cv2.flip(frame, 1)

    # 转换为 RGB 格式（MediaPipe 需要）
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 处理图像，获取手部关键点信息
    results = hands.process(frame_rgb)

    # 如果检测到手部关键点
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 遍历每个关键点，获取其坐标
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # 将相对坐标转换为像素坐标
                h, w, c = frame.shape  # 获取图像尺寸
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                # 在图像上标记关键点
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                # 输出关键点 ID 和像素坐标
                print(f"ID: {idx}, X: {cx}, Y: {cy}")

            # 绘制手部关键点和骨架
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 计算当前帧的时间，并计算帧率
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)  # 帧率 = 1 / 时间差
    prev_time = curr_time  # 更新前一帧时间

    # 在图像上显示帧率
    cv2.putText(frame, f'FPS: {int(fps)}', (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 显示图像
    cv2.imshow("Hand Tracking", frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
