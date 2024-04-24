import cv2 as cv
import mediapipe as mp
import math
import numpy as np


class AutomatedTrainerGuide:

    def __init__(self, stage = None, counter = 0, color = None):
        self.stage = stage
        self.counter = counter
        self.color = color

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()


    def findLMPose(self, image, draw=True):
        RGBImg = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Detect keypoints
        self.results = self.pose.process(RGBImg)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(image, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS,
                                    self.mpDraw.DrawingSpec((0, 0, 245), 5, 2),
                                    self.mpDraw.DrawingSpec((0, 240, 0), 2, 2))
                
        return image


    def getLMImagePosition(self, img, draw=True):
        # Find Position of Landmarks in Image and convert to pixels of image
        self.landmarkList = []
        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = img.shape
                # Get landmark position in image frame (the pixel value)
                landmarkWidth, landmarkHeight = int(landmark.x * width), int(landmark.y * height)
                # Append values of landmarkWidth and landmarkHeight to List
                self.landmarkList.append([id, landmarkWidth, landmarkHeight])

                if draw:
                    cv.circle(img, (landmarkWidth, landmarkHeight), 5, (255, 0, 0), -1)

        return self.landmarkList


    def findAngle(self, img, point1, point2, point3, draw=True):
        # Get width and height of landmarks from Landmark list, i.e. the x and y values only
        # For values of width and height at given landmark points
        width1, height1 = self.landmarkList[point1][1:]
        width2, height2 = self.landmarkList[point2][1:]
        width3, height3 = self.landmarkList[point3][1:]

        # Calculate the angle between the 3 points at the middle point
        self.angle = math.degrees(math.atan2(height3 - height2, width3 - width2)
                            - math.atan2(height1 - height2, width1 - width2))

        # Keeping the angle positive
        if self.angle < 0:
            self.angle += 360

        if draw:
            cv.line(img, (width1, height1), (width2, height2), (255, 255, 255), 3)
            cv.line(img, (width3, height3), (width2, height2), (255, 255, 255), 3)

            # Draw point1 outline and inner circle
            cv.circle(img, (width1, height1), 10, (0, 0, 240), cv.FILLED)
            cv.circle(img, (width1, height1), 15, (0, 0, 240), 2)

            # Draw point2 outline and inner circle
            cv.circle(img, (width2, height2), 10, (0, 0, 240), cv.FILLED)
            cv.circle(img, (width2, height2), 15, (0, 0, 240), 2)

            # Draw point3 outline and inner circle
            cv.circle(img, (width3, height3), 10, (0, 0, 240), cv.FILLED)
            cv.circle(img, (width3, height3), 15, (0, 0, 240), 2)

            pt = tuple(np.multiply(point2, [640, 480]).astype(int))
            # Show angle value
            cv.putText(img, str(int(self.angle)), (width2 - 50, height2 + 50),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
        return self.angle


    def curlCountlogic(self):
        if self.angle > 310:
            if self.stage == 'down':
                self.stage = 'up'
                self.counter += 1

        if self.angle < 210:
            self.stage = 'down'

        return self.counter, self.stage


    def progressBarlogic(self, percentage):
        if percentage == 100:
            self.color = (50, 50, 50)
            
        if percentage == 0:
            self.color = (0, 255, 0)

        return self.color


    def showCurlcount(self, img):
        # SHOW CURL REP DATA OR COUNT
        cv.rectangle(img, (970, 270), (1184, 480), (50, 50, 50), cv.FILLED)
        cv.putText(img, str(self.counter), (970, 420), cv.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)

    '''
    def showCurlstage(self, img):
        # SHOW CURL stage DATA
        cv.putText(img, 'STAGE', (65, 12), cv.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1)
        cv.putText(img, str(self.stage), (60, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    '''


    def progressBar(self, img, bar):
        # Show progress rectangle
        cv.rectangle(img, (796, 120), (836, 600), self.color, 2)
        cv.rectangle(img, (796, int(bar)), (836, 600), self.color, cv.FILLED)

        # Show percentage value
        # cv.putText(img, f'{int(percentage)}%', (1100, 75), cv.FONT_HERSHEY_SIMPLEX,
                # 1, (255, 255, 255), 2)
