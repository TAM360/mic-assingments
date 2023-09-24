import cv2 as cv


# Blur/smoothing is applied on the middle pixel of the
# kernel matrix with respect to its surrounding pixel
# values

def rescaleImage(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


img = cv.imread("./wallpaperflare.com_wallpaper.jpg")
img = rescaleImage(img, 0.25)

# averaging:
# computes the middle pixel's value by taking the
# average of the surrounding pixels.
# kernel's size is directly proportional to the
# intensity of blur
avg_blur = cv.blur(img, (5, 5))

# gausian:
# assigns weight to the surrounding pixels, computes the
# the average, and sets it in the middle pixel.
# yields less blurry (but more natual) image compare to
# averaging technique.
gauss_blur = cv.GaussianBlur(img, (5, 5), 0)

# median:
# computes median of the surrounding pixels and
# assigns it to the middle pixel.
# generally, it is considered to be more effective
# in reducing noise (salt and pepper) relative to the
# averaging and gaussing techniques.
# it also takes a single integer value as kernel, not
# tuple
median_blur = cv.medianBlur(img, 5)

# bilateral:
# most advance technique. blurs the image while maintaing
# the edges
bilateral_blur = cv.bilateralFilter(img, 5, 15, 15)

# cv.imshow("original", img)
# cv.imshow("avg-blur", avg_blur)
cv.imshow("gaussian-blur", gauss_blur)
cv.imshow("median-blur", median_blur)
cv.imshow("bilateral-blur", bilateral_blur)
cv.waitKey(0)