import cv2
import numpy as np
#创建画布
canvas = np.zeros((800, 800, 3), dtype=np.uint8)

#画图
center = (300, 300)
cv2.circle(canvas, center, 100, (255, 0, 0), -1)

for j in range(7):
    k = j*50
    top_left = (40+k, 500)
    bottom_right = (20+k, 550)
    cv2.rectangle(canvas, top_left, bottom_right, (0, 255, 0), -1)

cv2.ellipse(canvas, (600, 200), (150, 100), 0, 0, 360, (0, 255, 255), -1)

points = np.array([[100, 100], [200, 50], [300, 150], [250, 200]], np.int32).reshape((-1, 1, 2))
cv2.polylines(canvas, [points], True, (255, 255, 0), 2)

#边缘检测与轮廓提取
gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
canvas_= canvas.copy()
canvas_ = cv2.drawContours(canvas_, contours, -1, (0, 0, 255), 2)
cv2.imwrite("edges.png", canvas_)

contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

canvas_1 = canvas.copy()
number = 0

cv2.imwrite('shapes.png', canvas)
# 标记轮廓
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.putText(canvas, f" {number}", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 0, 255), 2)
    number += 1



template = cv2.imread("template.png")
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
contours_1, hierarchy = cv2.findContours(template_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

canvas_gray = cv2.cvtColor(canvas_1, cv2.COLOR_BGR2GRAY)

# 找到轮廓
for i in contours_1:
    perimeter = cv2.arcLength(i, True)
    circularity_1 = 4 * np.pi * (cv2.contourArea(i[0]) / (perimeter * perimeter))

    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter != 0:
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            if circularity_1 +0.8 < circularity < circularity_1 +0.85:
                cv2.putText(canvas_1, f"ellipse", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.rectangle(canvas_1, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imwrite('matching.png', canvas_1)
cv2.waitKey(0)
cv2.destroyAllWindows()

