# 实现思路
## 1. 使用OpenCV几何图形绘制  
### 创建画布：  
800×800的白色背景
### 几何元素绘制：  
使用 cv.polylines() 绘制三角形边框  
使用 cv.fillPoly() 填充三角形内部  
使用 cv.circle() 绘制3个圆形  
使用 cv.rectangle() 绘制3个矩形  
使用 cv.ellipse() 绘制3个椭圆  
##  2. 边缘检测与计数
### 边缘检测  
- 灰度转换 `imgGray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)`  
- Canny边缘检测`imgCanny=cv.Canny(imgGray,50,150)`
### 使用轮廓检测统计图形数量、在原图上标注检测到的轮廓
- 二值化处理  
- 查找所有轮廓cv.findContours() 
- 轮廓去重  
将因为Canny边缘检测产生的双重边缘等原因产生的同一个图形的重复累计的轮廓去重。   
for循环计算每个轮廓的质心、面积，把面积小的过滤掉，is_duplicate检查轮廓是否重复：计算质心，质心距离过近则取面积大的那个为该图形的轮廓；轮廓不重复就添加到unique_contours中。
- 标注轮廓  
for循环通过轮廓近似后按顶点数和比例确定unique_contours中每个轮廓的形状，同时统计图形数量；在图上把轮廓和计数标出来。  
## 3.模板匹配
- `h,w,l= template.shape` 取出模板图像长宽
- cv.matchTemplate() 模板匹配
- 返回图像中最匹配的位置，确定左上角的坐标，将匹配位置标注在图像上
## 4.姿态估计与动作分析
- 获取视频参数  
- 逐帧处理视频  
`results = pose.process(imgRGB)`返回结果  
`mpDraw.draw_landmarks()`绘制骨架  
遍历每帧每个关键点计入csv中，绘制关键点





