import cv2 as cv

video = cv.VideoCapture("./saving-project.gif")

def rescaleImage(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

while True:
    isTrue, frame = video.read()
    cv.imshow("video", rescaleImage(frame, 0.25))

    if cv.waitKey(20) and 0xFF == ord("d"):
        break


video.release()
cv.destroyAllWindows()
cv.waitKey(0)