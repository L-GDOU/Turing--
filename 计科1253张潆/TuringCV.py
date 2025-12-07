import numpy as np
import cv2 as cv
import csv
import time
import mediapipe as mp

#1.绘制几何图形
img=np.ones((800,800,3),dtype=np.uint8) * 255
points = np.array([[400,400], [300,525], [500,525]], np.int32)
cv.polylines(img, [points], True, (0, 255, 255), 5)
cv.fillPoly(img, [points], (224, 255, 255))
cv.circle(img,(275,700), 30, (185, 218, 255), -1)
cv.circle(img,(710,90), 67,(193, 182, 255) , -1)
cv.circle(img,(475,700), 67, (128, 128, 240), -1)
cv.rectangle(img,(21,27),(108,225),(122, 160, 255),-1)
cv.rectangle(img,(225,60),(600,175),(99,27,11),3)
cv.rectangle(img,(275,225),(679,579),(67,67,12),5)
cv.ellipse(img,(300,125),(60,35),0,0,360,(250,206,135),-1)
cv.ellipse(img,(100,700),(60,75),0,0,360,(0,0,255),2)
cv.ellipse(img,(100,450),(67,81),0,0,360,(144, 238, 144),-1)
img1=img.copy()
cv.imwrite('shapes.png',img)

#2.边缘检测与计数
#用Canny边缘检测
imgGray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
imgCanny=cv.Canny(imgGray,50,150)
cv.imwrite('edges.png',imgCanny)
#轮廓检测
ret,binary=cv.threshold(imgCanny,127,255,cv.THRESH_BINARY)
binary_copy=binary.copy()
contours,hierarchy=cv.findContours(binary_copy,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
unique_contours = []#轮廓去重
min_distance = 50
for i, contour in enumerate(contours):
    area = cv.contourArea(contour)
    if area < 2500:
        continue
    M = cv.moments(contour)
    if M["m00"] == 0:
        continue
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    # 检查是否与已有轮廓重复
    is_duplicate = False
    for j, unique_contour in enumerate(unique_contours):
        M2 = cv.moments(unique_contour)
        if M2["m00"] == 0:
            continue
        cx2 = int(M2["m10"] / M2["m00"])
        cy2 = int(M2["m01"] / M2["m00"])
        # 如果质心距离近，认为是同一个图形的不同部分
        distance = np.sqrt((cx - cx2) ** 2 + (cy - cy2) ** 2)
        if distance < min_distance:
            area2 = cv.contourArea(unique_contour)
            if area > area2:
                unique_contours[j] = contour
            is_duplicate = True
            break
    if not is_duplicate:
        unique_contours.append(contour)
contours = unique_contours
num=0#计数、标注每个轮廓
for contour in contours:
    area=cv.contourArea(contour)
    if area<2500:
        continue
    # 轮廓近似
    epsilon=0.02*cv.arcLength(contour,True)
    approx=cv.approxPolyDP(contour,epsilon,True)
    # 确定形状
    vertices=len(approx)
    if vertices==3:
        shape="Triangle"
    elif vertices==4:
        x,y,w,h=cv.boundingRect(approx)
        aspect_ratio=float(w)/h
        if 0.95<=aspect_ratio<=1.05:
            shape="Square"
        else:
            shape="Rectangle"
    else:
        (x,y),(width,height),angle=cv.minAreaRect(contour)
        aspect_ratio=min(width,height)/max(width,height)
        if aspect_ratio>0.9:
            shape="Circle"
        else:
            shape="Ellipse"
    M=cv.moments(contour)
    if M["m00"]!=0:
        cx=int(M["m10"]/M["m00"])
        cy=int(M["m01"]/M["m00"])
        # 标注检测到的轮廓
        cv.putText(img,shape,(cx-45,cy+3),
                    cv.FONT_HERSHEY_SIMPLEX,1.0,(0, 0, 0),2)
    # 绘制轮廓
    cv.drawContours(img,[contour], -1,(0, 0, 0),2)
    num+=1
s=f'num:{num}'
cv.putText(img,s,(670,700),cv.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,0),1,cv.LINE_AA)
cv.imwrite('contours.png',img)

#3.模板匹配
template=cv.imread('template.png')
template = template.astype(np.uint8)
h,w,l= template.shape
res = cv.matchTemplate(img1, template, cv.TM_SQDIFF_NORMED)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
top_left = min_loc
bottom_right = (top_left[0]+w, top_left[1] +h)
cv.rectangle(img1, top_left, bottom_right,(255, 47, 20), 2)
cv.imwrite('matching.png',img1)

#4.姿态估计与动作分析
pTime=0
cap = cv.VideoCapture('p.mp4')
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)
fourcc = cv.VideoWriter_fourcc(*'avc1')
output_path='pose.mp4'
out=cv.VideoWriter(output_path,fourcc,fps,(frame_width, frame_height))
mpPose=mp.solutions.pose
pose=mpPose.Pose()
mpDraw =mp.solutions.drawing_utils
csv_file = open('repl.csv', 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
header = ['frame_number', 'landmark_id', 'x_pixel', 'y_pixel', 'z_normalized', 'visibility']
csv_writer.writerow(header)
frame_number=0
while True:# 逐帧处理视频
    success,img=cap.read()
    frame_number += 1
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results=pose.process(imgRGB)#返回检测结果
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)#绘制姿态
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c=img.shape
            cx,cy=int(lm.x * w),int(lm.y * h)
            csv_writer.writerow([
                frame_number,
                id,
                cx,
                cy,
                round(lm.z,4),
                round(lm.visibility,4)
            ])#写入csv
            cv.circle(img,(cx, cy),5,(255,0,0),cv.FILLED)
    out.write(img)

