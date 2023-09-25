import cv2 as cv
import numpy as np


def rescaleImage(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


img_99_28 = rescaleImage(cv.imread("Assignment_1_MIC/msds23006/99_28.jpg"))

kernel = (3, 3)
threshold_val, thresh = cv.threshold(
    img_99_28, 175, 255, cv.THRESH_BINARY
)  # simple thresholding

# histogram equalization
hist_equal = cv.equalizeHist(
    cv.cvtColor(img_99_28, cv.COLOR_BGR2GRAY)
)  # for improving contrast
median_blur = cv.medianBlur(hist_equal, 7, 0)
bilateral_blur = cv.bilateralFilter(hist_equal, 7, 15, 15)

# Contrast Limited Adaptive Histogram Equalization
# Linear contrast adjustmen
clahe = cv.createCLAHE(clipLimit=40)
clahe_img = clahe.apply(cv.cvtColor(img_99_28, cv.COLOR_BGR2GRAY))
clahe_img_median_blur = cv.medianBlur(clahe_img, 7, 0)

# denoising
fast_n_means_denoising = cv.fastNlMeansDenoising(img_99_28, None, 10.0, 7, 21)
fast_n_means_denoising_multi = cv.fastNlMeansDenoisingMulti(
    img_99_28, 2, 5, None, 4, 7, 35
)

# image filtering with 2D convolution
img_convolution_2d = cv.filter2D(hist_equal, -1, np.ones((5, 5), np.float32) / 25)

# cv.imshow("original 99_28.jpg", img_99_28)
# cv.imshow("simple thresholding 99_28.py", thresh)
cv.imshow("histogram equalization 99_28.jpg", hist_equal)
cv.imshow("histogram equalization with median blur 99_28.jpg", median_blur)
# cv.imshow("histogram equalization with bilateral blur 99_28.jpg", bilateral_blur)
# cv.imshow("fast_n_means_denoising 99_28.jpg", fast_n_means_denoising)
# cv.imshow(
#     "fast_n_means_denoising_multi 99_28.jpg",
#     fast_n_means_denoising_multi
# )
# cv.imshow("2d convolution 99_28.jpg", img_convolution_2d)
# cv.imshow("CLAHE 99_28.jpg", clahe_img)
# cv.imshow("CLAHE median blur", clahe_img_median_blur)

cv.waitKey(0)
