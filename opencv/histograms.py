import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def rescaleImage(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


img = cv.imread("./wallpaperflare.com_wallpaper.jpg")
img = rescaleImage(img, 0.25)
gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blank = np.zeros(img.shape[:2], dtype="uint8")
circle = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 150, 255, -1)
mask = cv.bitwise_and(gray_scale, gray_scale, mask=circle)

# gray scale histogram
gray_histogram = cv.calcHist([gray_scale], [0], None, [256], [0, 256])
gray_histogram_with_mask = cv.calcHist([gray_scale], [0], mask, [256], [0, 256])

plt.figure()
plt.title("gray scale histogram")
plt.xlabel("bins")
plt.ylabel("# of pixels")
plt.xlim(0, 256)

# cv.imshow("gray", gray_scale)
# cv.waitKey(0)

plt.plot(gray_histogram)
plt.plot(gray_histogram_with_mask)
plt.show()
