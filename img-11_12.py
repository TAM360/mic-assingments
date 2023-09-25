import cv2 as cv
import numpy as np

def rescaleImage(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img_11_12 = rescaleImage(cv.imread("Assignment_1_MIC/msds23006/11_22.jpg"))

# Morphological operations
kernel = (3, 3) # np.ones((3, 3), np.uint8)
eroded = cv.erode(img_11_12, kernel, iterations=1)
dilated = cv.dilate(img_11_12, kernel, iterations=1)
opening = cv.morphologyEx(img_11_12, cv.MORPH_OPEN, kernel) # openining -> erosion followed by dilation
closing = cv.morphologyEx(img_11_12, cv.MORPH_CLOSE, kernel) # closing -> dilation followed by erosion

# blur
gaus_blur = cv.GaussianBlur(img_11_12, (5, 5), 0)
bilateral_blur = cv.bilateralFilter(img_11_12, 5, 15, 0)
median_blur = cv.medianBlur(img_11_12, 5)

cv.imshow("original ", img_11_12)
# cv.imshow("eroded img_11_12.jpg", eroded)
# cv.imshow("dilated img_11_12.jpg", dilated)
# cv.imshow("opening img_11_12.jpg", opening)
# cv.imshow("closing img_11_12.jpg", closing)

# cv.imshow("gaussian blur 11_12,jpg", gaus_blur)
# cv.imshow("bilateral blur 11_12.jpg", bilateral_blur)
cv.imshow("median blur 11_12.jpg", median_blur)

cv.waitKey(0)