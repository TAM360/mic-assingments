import cv2 as cv
import numpy as np


def rescaleImage(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


# translation -> shift im image along x and/or y axis
# -x -> left
# -y -> up
# +x -> right
# +y -> down
def translate(image, x, y):
    transMatrix = np.float32([[1, 0, x], [0, 1, y]])
    dim = (image.shape[1], image.shape[0])

    return cv.warpAffine(image, transMatrix, dim)


# rotation
def rotate(image, angle, rotationPoint=None):
    (height, width) = image.shape[:2]

    if rotationPoint == None:
        rotationPoint = (width // 21, height // 2)  # extract the central pt

    rotationMatrix = cv.getRotationMatrix2D(rotationPoint, angle, 1.0)
    dim = (width, height)
    return cv.warpAffine(image, rotationMatrix, dim)


# read an image
img = cv.imread("./wallpaperflare.com_wallpaper.jpg")
img = rescaleImage(img, 0.25)

# convert colored image into gray scale image
gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# translate image
translated = translate(gray_scale, 0, 200)

# rotated image
rotated = rotate(gray_scale, -45)

# image rescaling (without custom method)
# for image shrinking, use INTER_AREA
# for image enlarging, use either INTER_CUBIC or INTER_LINEAR
# INTER_LINEAR is simple. INTER_CUBIC is expensive but gives
# better results
resized_area = cv.resize(img, [550, 500], interpolation=cv.INTER_AREA)
resized_linear = cv.resize(img, [550, 500], interpolation=cv.INTER_LINEAR)
resized_cubic = cv.resize(img, [550, 500], interpolation=cv.INTER_CUBIC)

# flipping
flipX = cv.flip(gray_scale, 0)
flipY = cv.flip(gray_scale, 1)
flipXandY = cv.flip(gray_scale, -1)

cv.imshow("gray", gray_scale)
# cv.imshow("translated", translated)
# cv.imshow("rotated", rotated)
# cv.imshow("resized-area", resized_area)
# cv.imshow("resized-linear", resized_linear)
# cv.imshow("resized-cubic", resized_cubic)
# cv.imshow("flipped-along-x-axis", flipX)
# cv.imshow("flipped-along-y-axis", flipY)
# cv.imshow("flipped-along-y-and-x-axis", flipXandY)


cv.waitKey(0)
