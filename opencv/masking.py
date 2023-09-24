import cv2 as cv
import numpy as np

# Masking is used to removing certain region of an image
# in order to focus on the rest
# Masking involves bitwise operations

def rescaleImage(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


img = cv.imread("./wallpaperflare.com_wallpaper.jpg")
img = rescaleImage(img, 0.25)

# NOTE: mask dimensions must match image's size
blank = np.zeros(img.shape[:2], dtype="uint8")
mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 150, 255, -1)

masked_img = cv.bitwise_and(img, img, mask=mask)
# cv.imshow("original", img)
# cv.imshow("mask", mask)
cv.imshow("masked-image", masked_img)
cv.waitKey(0)
