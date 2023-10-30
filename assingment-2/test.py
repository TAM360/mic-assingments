import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage import io


class ImagePatcher(object):
    def __init__(self, image, dimensions):
        self.image = image
        self.dimensions = dimensions
        self.N, self.M = dimensions  # N = # of rows, M = # of columns
        self.num_rows, self.num_cols = self.get_patch_rows_and_columns()
        self.image_patch_list = []
        self.hist_list = np.empty((self.N * self.M, 256))

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

    def merge_image_patches(self, result):
        canvas = np.zeros_like(self.image, dtype=np.uint8)

        # TODO: investigate a solution for mapping colors to patches
        color_map = {
            0: (255, 0, 0),
            1: (0, 0, 255),
            2: (0, 255, 0),
            3: (247, 247, 0),
        }

        # for label in result.labels_:
        #     canvas = color_map[label]

        for row in range(self.N):
            for col in range(self.M):
                x_start = col * self.num_cols
                x_end = (col + 1) * self.num_cols
                y_start = row * self.num_rows
                y_end = (row + 1) * self.num_rows
                patch = self.image_patch_list[row][col]
                print("patch", patch)

                canvas[y_start:y_end, x_start:x_end] = patch

        return canvas

    def get_histograms(self):
        count = 0
        for row in range(self.N):
            for col in range(self.M):
                patch = self.image_patch_list[row][col]
                hist = cv.calcHist([patch], [0], None, [256], [0, 256])
                print(min(hist), max(hist), patch.shape)
                self.hist_list[count, :] = hist.reshape(256)
                count += 1

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


image_patch_1 = ImagePatcher(image_1, (5, 5))
image_patch_list_1 = image_patch_1.split_image("image-1")

# patch level histograms
image_patch_histograms_1 = image_patch_1.get_histograms()
# print(image_patch_histograms_1.shape)
# print(type(image_patch_histograms_1))


# k-means algo
k = 4
criteria = (cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv.KMEANS_RANDOM_CENTERS

kmeans = KMeans(
    n_clusters=k, init="random", random_state=0, n_init="auto"
)

kmeans_result = kmeans.fit(image_patch_histograms_1)
print(kmeans_result.labels_, len(kmeans_result.labels_))
result = image_patch_1.merge_image_patches(kmeans_result)
# super-pixel segmentation

# active contouring
# contour_image_1 = get_image_contour(image_1)
# contour_image_2 = get_image_contour(image_2)
# contour_image_3 = get_image_contour(image_3)
# contour_image_4 = get_image_contour(image_4)
# contour_image_7 = get_image_contour(image_7)

# otsu thresholding
# _, image_thresh_ostu_1 = cv.threshold(
#     image_1, 120, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
# )
# _, image_thresh_ostu_2 = cv.threshold(
#     image_2, 120, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
# )
# _, image_thresh_ostu_3 = cv.threshold(
#     image_3, 120, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
# )
# _, image_thresh_ostu_4 = cv.threshold(
#     image_4, 150, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
# )
# _, image_thresh_ostu_7 = cv.threshold(
#     image_7, 120, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
# )

# super-pixel thresholding
# segments = slic(image_1, n_segments=50, sigma=5, channel_axis=None)
# fig = plt.figure("Superpixels -- %d segments" % (100))
# ax = fig.add_subplot(1, 1, 1)
# ax.imshow(mark_boundaries(image_1, segments))
# plt.axis("off")
# plt.show()

# cv.imshow("active contour 1", con25tour_image_1)
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
