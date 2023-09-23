import cv2 as cv

# read an image
img = cv.imread("./Screenshot from 2023-09-17 14-32-02.png")

# display the image
cv.imshow("wallpaper", img)

# make display window wait for X milliseconds before closing
cv.waitKey(2000)

# read a video.
video = cv.VideoCapture("./saving-project.gif")

# capture each frame of the video and show it
# no direct way of displaying video
while True:
    isTrue, frame = video.read()
    cv.imshow("video", frame)

    if cv.waitKey(20) and 0xFF == ord("d"):
        break

video.release()
cv.destroyAllWindows()

cv.waitKey(0)