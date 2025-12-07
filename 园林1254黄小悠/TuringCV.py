import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import math

#Part1: 使用OpenCV几何图形绘制
canvas = np.ones((800, 800, 3), dtype=np.uint8) * 255
THICK = 3

#定义颜色
BLUE   = (90, 125, 175)
GREEN  = (120, 165, 145)
ORANGE = (215, 165, 110)
PURPLE = (165, 145, 195)
GRAY   = (140, 140, 140)

#绘制矩形
cv2.rectangle(canvas, (80, 80), (240, 200), BLUE, -1)
cv2.rectangle(canvas, (80, 240), (240, 360), BLUE, THICK)
cv2.rectangle(canvas, (560, 80), (720, 200), ORANGE, -1)
cv2.rectangle(canvas, (580, 630), (700, 740), GREEN, -1)

#绘制圆
cv2.circle(canvas, (160, 460), 60, GREEN, -1)
cv2.circle(canvas, (160, 620), 55, GREEN, THICK)
cv2.circle(canvas, (640, 310), 60, ORANGE, THICK)

#绘制椭圆
cv2.ellipse(canvas, (400, 140), (85, 45), 20, 0, 360, ORANGE, THICK)
cv2.ellipse(canvas, (400, 280), (75, 40), -15, 0, 360, ORANGE, -1)

#绘制多边形
triangle = np.array([[340, 360],[460, 360],[400, 500]])
cv2.drawContours(canvas, [triangle], 0, PURPLE, -1)

pentagon = np.array([[400, 560],[460, 600],[435, 660],[365, 660],[340, 600]])
cv2.drawContours(canvas, [pentagon], 0, PURPLE, THICK)

hexagon = np.array([[640, 460],[690, 500],[690, 560],[640, 600],[590, 560],[590, 500]])
cv2.drawContours(canvas, [hexagon], 0, GRAY, THICK)

cv2.imwrite("shapes.png", canvas)



#Part2: 边缘检测与计数
img = cv2.imread("shapes.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 80, 150)
cv2.imwrite("edges.png", edges)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_img = img.copy()
cv2.drawContours(contours_img, contours, -1, (0, 0, 0), 2)  #黑色轮廓
cv2.imwrite("contours.png", contours_img)

#输出检测到的图形数量
num_shapes = len(contours)
print("检测到的图形数量:", num_shapes)



#Part3: 模板匹配（匹配所有圆形并标注绿色框）
template = img[565:675, 105:215]  #左列描边圆模板
cv2.imwrite("template.png", template)

#模板匹配
result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.7
locations = np.where(result >= threshold)

h, w = template.shape[:2]
seen = set()  #去重重复匹配框
matching_img = img.copy()
for pt in zip(*locations[::-1]):
    x1, y1 = pt
    x2, y2 = x1 + w, y1 + h
    if (x1, y1, x2, y2) in seen:
        continue
    seen.add((x1, y1, x2, y2))
    cv2.rectangle(matching_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  #绿色标注

cv2.imwrite("matching.png", matching_img)



#Part4：姿态估计与动作分析（用MediaPipe）
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

#显式配置姿态模型参数
pose = mp_pose.Pose(
    static_image_mode=False,     #视频模式
    model_complexity=1,          #精度与速度折中
    smooth_landmarks=True,       #关键点时间平滑
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#打开视频
cap = cv2.VideoCapture("pose.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#初始化视频写入器
out = cv2.VideoWriter(
    "pose_out.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (W, H)
)

#数据记录变量
records = []              #保存所有关键点
frame_id = 0

prev_mid_hip = None       #上一帧髋部中点
speed_px = []             #相对速度（px/s）
height_px_list = []       #人体像素高度


#主循环:逐帧姿态估计
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #MediaPipe使用RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    #若检测到人体姿态
    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark

        #绘制骨架
        mp_draw.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        #髋部中点（像素坐标）

        l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

        mid_hip = np.array([
            (l_hip.x + r_hip.x) / 2 * W,
            (l_hip.y + r_hip.y) / 2 * H
        ])

        #转为像素速度(px/s)

        if prev_mid_hip is not None:
            dist_px = np.linalg.norm(mid_hip - prev_mid_hip)
            speed_px.append(dist_px * fps)

        prev_mid_hip = mid_hip

        #保存所有关键点坐标

        for idx, lm_i in enumerate(lm):
            records.append([
                frame_id,
                idx,
                lm_i.x,
                lm_i.y,
                lm_i.z
            ])

    #写入输出视频
    out.write(frame)
    frame_id += 1


#资源释放
cap.release()
out.release()

#保存关键点CSV
df = pd.DataFrame(
    records,
    columns=["frame", "landmark_id", "x", "y", "z"]
)
df.to_csv("repl.csv", index=False)

#输出平均相对速度(px/s)
avg_speed_px = np.mean(speed_px)
print(f"平均跑步相对速度（px/s）：{avg_speed_px:.2f}")
