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

cv.imshow("lab-to-bgr", bgr)
cv.waitKey(0)