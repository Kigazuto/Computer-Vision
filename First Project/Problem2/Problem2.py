import cv2
import numpy as np

def circle_detect(image):
    filter = cv2.pyrMeanShiftFiltering(image, 10, 100)
    gray = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 500, param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 1)
    cv2.imshow("circle_detect", image)

src = cv2.imread("F:\Pycharm\Computer Vision\Problem2\Mar.jpg")
cv2.imshow("source_image", src)
circle_detect(src)
cv2.waitKey(0)
cv2.destroyAllWindows()