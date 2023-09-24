import cv2 as cv
import numpy as np


def rescaleImage(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


blank = np.zeros((400, 400), dtype="uint8")
img = cv.imread("./wallpaperflare.com_wallpaper.jpg")
img = rescaleImage(img, 0.25)
rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)

# bitwise AND:
# returns intersection of 2 images
bitwise_and = cv.bitwise_and(circle, rectangle)

# bitwise OR:
# combines 2 images into one
bitwise_or = cv.bitwise_or(circle, rectangle)

# bitwise XOR
# removes intersecting pixels and returns the image
bitwise_xor = cv.bitwise_xor(circle, rectangle)

# bitwise NOT
bitwise_not = cv.bitwise_not(circle)

# cv.imshow("original", img)
# cv.imshow("rectangle", rectangle)
# cv.imshow("circle", circle)
cv.imshow("bitwise-and", bitwise_and)
cv.imshow("bitwise-or", bitwise_or)
cv.imshow("bitwise-xor", bitwise_xor)
cv.imshow("bitwise-not", bitwise_not)

cv.waitKey(0)
