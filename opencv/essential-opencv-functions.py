import cv2 as cv


def rescaleImage(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


# read an image
img = cv.imread("./wallpaperflare.com_wallpaper.jpg")
img = rescaleImage(img, 0.25)

# convert colored image into gray scale image
gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# blur image using gaussian method
# ksize -> kernel size (must be a vector of odd numers)
#          higher value will yield more blur image

blur = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)

# edge cascade
canny = cv.Canny(img, 125, 175)
# to rduce the # of edges, use blur image
canny_blur = cv.Canny(blur, 125, 175)

# image dilation
# makes edges in canny image thick or thin
dilated = cv.dilate(canny, (7, 7), iterations=1)

# erosion: removes edges in N iterations
eroded = cv.erode(dilated, (3, 3), iterations=3)

# cropping
cropped = gray_scale[50:400, 0:800]

cv.imshow("gray", gray_scale)
# cv.imshow("blur", blur)
# cv.imshow("canny", canny)
# cv.imshow("canny-blur", canny_blur)
# cv.imshow("dilated", dilated)
# cv.imshow("eroded", eroded)
cv.imshow("cropped", cropped)

cv.waitKey(0)
