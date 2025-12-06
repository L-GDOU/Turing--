import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
path_1=Path(r"E:\python\opencv\inoe2.png")


# 创建一个白色背景
img=np.full((800,800,3),255,np.uint8)
# 绘制一个紫色圆
cv.circle(img,(160,160),150,(255,0,128),10)
# 绘制一个橙色圆
cv.circle(img,(650,150),100,(0,128,255),10)
# 绘制一个蓝色矩形
cv.rectangle(img,(50,400),(300,550),(256,0,0),10)
# 绘制一个蓝色实心正方形
cv.rectangle(img,(50,600),(150,700),(256,0,0),-1)
# 绘制一个红色椭圆
cv.ellipse(img,(450,500),(100,50),45,0,360,(0,0,255),10)
# 绘制一个黑色椭圆
cv.ellipse(img,(650,400),(100,20),90,0,360,(0,0,0),10)
#绘制一个水蓝色五边形
pts_1 = np.array([[400, 50], [450, 100], [430, 170], [370, 170], [350, 100]], np.int32)
cv.polylines(img, [pts_1],True,(256,256,0), thickness=10, lineType=8, shift=0)
#绘制一个黄色三角形
pts_2 = np.array([[400, 300], [600, 300], [500, 150]], np.int32)
cv.polylines(img, [pts_2],True,(0,256,256), thickness=10, lineType=8, shift=0)
#绘制一个品红色梯形
pts_3 = np.array([[600, 600], [700, 600], [750, 700], [550, 700]], np.int32)
cv.polylines(img, [pts_3],True,(256,0,256), thickness=10, lineType=8, shift=0)
# 绘制草绿色的填充三角形
pts_4 = np.array([[300, 600], [400, 700], [200, 750]], np.int32)
cv.fillPoly(img, [pts_4], (128,255,0), lineType=8, shift=0, offset=(0, 0))
# 原始几何图形
cv.imwrite("shapes.png",img)


# 预处理（转为灰度图，使用均值滤波）
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
blur=cv.blur(gray,(5,5))
# 对预处理的图像进行Canny边缘检测
img_edges=cv.Canny(img,0,100)
# 边缘检测结果
cv.imwrite("edges.png",img_edges)


# 闭运算，防止边缘检测后的图像不闭合
kernel=np.ones((3,3),np.uint8)
img_close=cv.morphologyEx(img_edges,cv.MORPH_CLOSE,kernel)
# 对闭运算的结果进行轮廓检测
contours,hierarchy=cv.findContours(img_close,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
# 绘制轮廓（绿色）
img_contours=cv.drawContours(img.copy(),contours,-1,(0, 255, 0),2)
# 统计图形数量
num=len(contours)
print(f"图形数量{num}")
# 轮廓检测结果
cv.imwrite("contours.png",img_contours)


# 截取模板
template=img[0:320,0:320]
# 模板图形
cv.imwrite("template.png",template)

# 获取模板的长宽
h,w=template.shape[:2]
# 进行模板匹配
matchTemplate=cv.matchTemplate(img,template,cv.TM_SQDIFF)
A,B,C,D= cv.minMaxLoc(matchTemplate)
top_left=C
# 绘制匹配结果
bottom_right = (top_left[0]+w, top_left[1]+h)
cv.rectangle(img,top_left,bottom_right,(0,255,0),5)
# 保存匹配结果
cv.imwrite("matching.png",img)