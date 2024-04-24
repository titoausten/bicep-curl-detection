import cv2 as cv
import numpy as np


def curlCount(cap, curlCounter, appBackground):
    while cap.isOpened():
        ret, image = cap.read()
        image = image.copy()
        image = cv.resize(image, (640, 480))

        # FIND POSE
        image = curlCounter.findLMPose(image, False)

        # FIND POSITION
        landmarklist = curlCounter.getLMImagePosition(image, False)

        if len(landmarklist) != 0:
            angle = curlCounter.findAngle(image, 11, 13, 15)

            percentage = np.interp(angle, (210, 310), (0, 100))
            bar = np.interp(angle, (220, 310), (600, 120))

            counter, stage = curlCounter.curlCountlogic()
            color = curlCounter.progressBarlogic(percentage)
            print(counter, stage, color)

            curlCounter.progressBar(appBackground, bar)
            curlCounter.showCurlcount(appBackground)

        appBackground[120:120+480,106:106+640] = image
        cv.imshow('Bicep Curl Counter', appBackground)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
