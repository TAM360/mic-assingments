import cv2 as cv
import numpy as np


def rescaleImage(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


img = cv.imread("./wallpaperflare.com_wallpaper.jpg")
img = rescaleImage(img, 0.25)
gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny = cv.Canny(gray_scale, 125, 125)
blur = cv.GaussianBlur(gray_scale, (5, 5), cv.BORDER_DEFAULT)

# thresholding
ret, threshold = cv.threshold(gray_scale, 125, 125, cv.THRESH_BINARY)

# image contours
# contours are defined as continous lines (curves) along the
# boundary of an object, having same color/intensity
# Note: For better accuracy use binary images (canny or threshold)
# contours_list is an array containing co-ordinates of all the
# contours that are present in an image
# hierachies shows hiearchal representation of the contours
# e.g rectable containing square, which in turn contains a circle
contours_list, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
# contours_list, hierarchies = cv.findContours(threshold, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

print("contours_list => ", contours_list, len(contours_list))
print("hierarchy => ", hierarchies, len(hierarchies))

# contour visualization using numpy
blank_img = np.zeros(img.shape, dtype="uint8")
cv.drawContours(blank_img, contours_list, -1, (0,0,255), 1)

cv.imshow("gray", gray_scale)
# cv.imshow("blur", blur)
# cv.imshow("canny edges", canny)
# cv.imshow("threshold", threshold)
cv.imshow("contour visualization", blank_img)
cv.waitKey(0)
