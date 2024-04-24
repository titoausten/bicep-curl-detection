import cv2 as cv
import BicepCurlTrainer
import numpy as np
from utils import curlCount


def main():
    cam_feed = input("Enter video feed number or video filepath: ")
    if cam_feed == '0':
        curlCounter = BicepCurlTrainer.AutomatedTrainerGuide()
        cap = cv.VideoCapture(int(cam_feed))
        cap.set(3, 640)
        cap.set(4, 480)
        appBackground = cv.imread("../data/bicepUIbg.png")
        curlCount(cap, curlCounter, appBackground)

    else:
        curlCounter = BicepCurlTrainer.AutomatedTrainerGuide()
        cap = cv.VideoCapture(cam_feed)
        cap.set(3, 640)
        cap.set(4, 480)
        appBackground = cv.imread("../data/bicepUIbg.png")
        curlCount(cap, curlCounter, appBackground)


if __name__ == "__main__":
    main()
