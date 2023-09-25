import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# thresholding:
# convert an image into it's binary representation using
# a thresholding value. A pixel < threhold will be become
# 0 (black) and a pixel >= threhold will become a 1 (white)


def rescaleImage(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


img = cv.imread("./wallpaperflare.com_wallpaper.jpg")
img = rescaleImage(img, 0.25)
gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# simple thresholding (with 150 as the threshold value)
threshold, thresh = cv.threshold(gray_scale, 100, 255, cv.THRESH_BINARY)

# adaptive thresholding
# adaptive_threshold = cv.adaptiveThreshold(
#     gray_scale, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 3
# )
adaptive_threshold = cv.adaptiveThreshold(
    gray_scale, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 3
)

cv.imshow("simple-thresholding", thresh)
cv.imshow("adaptive-thresholding", adaptive_threshold)
cv.waitKey(0)
