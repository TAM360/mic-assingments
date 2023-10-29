import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class ImagePatcher(object):
    def __init__(self, image, dimensions):
        self.image = image
        self.dimensions = dimensions
        self.N, self.M = dimensions  # N = # of rows, M = # of columns
        self.num_rows, self.num_cols = self.get_patch_rows_and_columns()
        self.image_patch_list = []
        self.hist_list = []

    def get_patch_rows_and_columns(self):
        height, width = self.image.shape

        # Calculate the number of rows and columns for a patch
        num_rows = height // self.N
        num_cols = width // self.M

        return (num_rows, num_cols)

    def split_image(self, dir_name):
        # Iterate through the large image and divide it into smaller sub-images
        for row in range(self.N):
            temp = []
            for col in range(self.M):
                # Calculate the coordinates for each sub-image
                x_start = col * self.num_cols
                x_end = (col + 1) * self.num_cols
                y_start = row * self.num_rows
                y_end = (row + 1) * self.num_rows

                sub_image = self.image[y_start:y_end, x_start:x_end].astype(np.float32)
                temp.append(sub_image)
                cv.imwrite(
                    f"./results/{dir_name}/image-patches/image-patch-{row + 1}-{col + 1}.jpg",
                    sub_image,
                )

            self.image_patch_list.append(temp)

        return self.image_patch_list

    def merge_image_patches(self, canvas_size):
        height, width = canvas_size
        canvas = np.zeros((height, width), dtype=np.uint8)

        for row in range(self.N):
            for col in range(self.M):
                x_start = col * self.num_cols
                x_end = (col + 1) * self.num_cols
                y_start = row * self.num_rows
                y_end = (row + 1) * self.num_rows
                patch = self.image_patch_list[row][col]

    def get_histograms(self):
        for row in range(self.N):
            temp = []
            for col in range(self.M):
                patch = self.image_patch_list[row][col]
                hist = cv.calcHist([patch], [0], None, [256], [0, 256])
                temp.append(hist)

            self.hist_list.append(temp)

        return self.hist_list


def get_image_contour(image, a=125, b=125):
    canny = cv.Canny(image, a, b)
    contours_list, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    blank_img = np.zeros(image.shape, dtype="uint8")
    cv.drawContours(blank_img, contours_list, -1, (255, 255, 255), 1)

    return blank_img


image_1 = cv.imread("./Dataset/1.jpg", 0)
image_2 = cv.imread("./Dataset/2.jpg", 0)
image_3 = cv.imread("./Dataset/3.jpg", 0)
image_4 = cv.imread("./Dataset/4.jpg", 0)
image_7 = cv.imread("./Dataset/7.jpg", 0)

image_1 = cv.resize(image_1, (1400, 900))
image_2 = cv.resize(image_2, (1400, 900))
image_3 = cv.resize(image_3, (1400, 900))
image_4 = cv.resize(image_4, (1400, 900))
image_7 = cv.resize(image_7, (1400, 900))

# image_patch_1 = ImagePatcher(image_1, (5, 5))
# image_patch_list_1 = image_patch_1.split_image("image-1")

# image_patch_histograms_1 = image_patch_1.get_histograms()

# for idx in range(5):
#     plt.figure()
#     plt.plot(image_patch_histograms_1[0][idx])

# plt.show()

# k means algo
# k = 4
# criteria = (cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# flags = cv.KMEANS_RANDOM_CENTERS
# compactness, labels, centers = cv.kmeans(
#     image_patch_histograms_1[0][0], 2, None, criteria, 10, flags
# )

# kmeans = KMeans(n_clusters=4, init="random", random_state=0, n_init="auto").fit(
#     image_patch_histograms_1[0][0]
# )
# print(kmeans.cluster_centers_)
# plt.show()

# super-pixel segmentation

# active contouring
contour_image_1 = get_image_contour(image_1)
contour_image_2 = get_image_contour(image_2)
contour_image_3 = get_image_contour(image_3)
contour_image_4 = get_image_contour(image_4)
contour_image_7 = get_image_contour(image_7)

# otsu thresholding
_, image_thresh_ostu_1 = cv.threshold(
    image_1, 120, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
)
_, image_thresh_ostu_2 = cv.threshold(
    image_2, 120, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
)
_, image_thresh_ostu_3 = cv.threshold(
    image_3, 120, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
)
_, image_thresh_ostu_4 = cv.threshold(
    image_4, 150, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
)
_, image_thresh_ostu_7 = cv.threshold(
    image_7, 120, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
)

# cv.imshow("active contour 1", contour_image_1)
# cv.imshow("active contour 2", contour_image_2)
# cv.imshow("active contour 3", contour_image_3)
# cv.imshow("active contour 4", contour_image_4)
# cv.imshow("active contour 7", contour_image_7)

# cv.imshow("image 1 otsu threshold", image_thresh_ostu_1)
# cv.imshow("image 2 otsu threshold", image_thresh_ostu_2)
# cv.imshow("image 3 otsu threshold", image_thresh_ostu_3)
# cv.imshow("image 4 otsu threshold", image_thresh_ostu_4)
# cv.imshow("image 7 otsu threshold", image_thresh_ostu_7)
cv.waitKey(0)
cv.destroyAllWindows()
