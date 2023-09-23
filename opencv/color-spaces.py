# color spaces refers to the representation of pixel arrays
# in different color schemes. RGB, and Gray scale are few of
# these examples

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def rescaleImage(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


img = cv.imread("./wallpaperflare.com_wallpaper.jpg")
img = rescaleImage(img, 0.25)
gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# gray to HSV color schem conversion
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# gray to LAB (l*a*b)
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

# RGB color space (different from BGR)
# NOTE: OpenCV uses BGR format to under stand the color scheme
# plt.imshow(img)
# plt.show()
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# HSV to BGR
bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

# LAB to BGR
bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

cv.imshow("original", img)
# cv.imshow("gray", gray_scale)
# cv.imshow("hsv", hsv)
# cv.imshow("lab", lab)
# cv.imshow("rgb", rgb)
# cv.imshow("hsv-to-bgr", bgr)
cv.imshow("lab-to-bgr", bgr)
cv.waitKey(0)