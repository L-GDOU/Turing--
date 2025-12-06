import cv2
import numpy as np
#创建画布 bu=shapes，后期才看到命名规范aa
bu=np.ones((800, 800, 3), dtype=np.uint8)*255
#画图
cv2.rectangle(bu, (0, 0), (150, 150), (11, 183, 99), -1)#正方1
cv2.rectangle(bu, (180,0), (300,120), (111, 222, 0), 2)#正方2
cv2.rectangle(bu,(350,0),(500,100), (112, 113, 0), 3)#长方形3
cv2.circle(bu,(630,200),150,(211,98,5),4)#圆形4
cv2.circle(bu,(400,200),66,(44,155,66),5)#圆形5
cv2.circle(bu,(100,200),30,(168,98,5),6)#圆形6
cv2.ellipse(bu,(125, 725),(120, 70),0,0,360,(0,110,88),7)#椭圆7
cv2.ellipse(bu,(110, 580),(100,60),0,0,360,(114,114,0),8)#椭圆8
cv2.ellipse(bu,(200,360),(130,50),45,0,360,(0,155,155),9)#椭圆9
ox,oy=600,600#设置图案中心（着实想不到什么图案了，乱画中……
w,h=200,390#图案宽和高
pts=np.array([
    [ox,oy-h//2],#上点
    [ox,oy+h//2],#下点
    [ox-w//2,oy],#左点
    [ox+w//2,oy]],np.int32).reshape((4,1,2))
cv2.polylines(bu, [pts], True, (0,150,255), 10)#乱画10（图案是菱形调换描点顺序的结果
cv2.imshow("shapes", bu)
cv2.waitKey(0)
cv2.imwrite("shapes.png",bu)


#2.边缘检测与计数
gray=cv2.cvtColor(bu,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray,(5,5),0)
_,heibai=cv2.threshold(gray,180,255,cv2.THRESH_BINARY_INV)#heibai在图形匹配中也得用
#查找轮廓
contours,hierarchy = cv2.findContours(heibai,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print("检测到的图形数量 =",len(contours))
edges=bu.copy()
for i,cnt in enumerate(contours):
    cv2.drawContours(edges,[cnt],-1,(0,0,255),5)
cv2.imshow("edges",edges)
cv2.waitKey(0)
cv2.imwrite("edges.png",edges)


#写标签
nb=bu.copy()#number是nb也没错,因为contours经常在代码里面出现，怕搞混所以先将图片叫做nb
for i,cnt in enumerate(contours):
    M=cv2.moments(cnt)
    if M["m00"]!=0:
        cx=int(M["m10"]/M["m00"])
        cy=int(M["m01"]/M["m00"])
        # 在图上画编号
        cv2.putText(nb, str(i+1),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2)
cv2.imshow("contours",nb)
cv2.imwrite("contours.png",nb)
cv2.waitKey(0)


#截图
template=bu[160:240,60:160]
cv2.imshow("template",template)#显示框选出来的图案
cv2.imwrite("template.png",template)
cv2.waitKey(0)

#匹配图形
gtemplate=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)#gtemplate是template的灰度图
_,hbtemplate=cv2.threshold(gtemplate,180,255,cv2.THRESH_BINARY_INV)#灰度图二值化
#此时需要匹配轮廓
ytcnt,_= cv2.findContours(heibai, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#原图（黑白heibai）的轮廓
mbcnt,_= cv2.findContours(hbtemplate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#模板图的轮廓
template_cnt=mbcnt[0]
matching=bu.copy()#复制份原图用于画相似图形的框框
for cnt in ytcnt:
    score=cv2.matchShapes(template_cnt, cnt, cv2.CONTOURS_MATCH_I1, 0.0)
    if score<0.02:  # 0.05 越小越严格，越大越宽松
        x,y,w,h=cv2.boundingRect(cnt)
        cv2.rectangle(matching,(x,y),(x+w,y+h),(0,0,255),3)
cv2.imshow("matching",matching)
cv2.waitKey(0)
cv2.imwrite("matching.png",matching)

