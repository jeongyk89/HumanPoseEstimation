"""Import Stuff"""
import cv2
import mediapipe as mp
import math
import numpy as np
from time import time
import csv

"""Set up mediapipe"""
mp_drawing = mp.solutions.drawing_utils     # Drawing helpers
mp_pose = mp.solutions.pose                 # Mediapipe Solutions
"""Set up OpenCV"""
cap = cv2.VideoCapture(0)
"""Set up time keeping"""
ledger = []


def time_keeping(pose, time_taken):
    global ledger
    ledger.append([pose, round(time_taken, 4)])
    return ledger


# Finding the angle between 3 points
def calculate_angle(a, b, c):
    x1, y1, z1 = a
    x2, y2, z2 = b
    x3, y3, z3 = c
    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360
    # Return the calculated angle.
    return angle

    # a = np.array(a)  # Startpoint
    # b = np.array(b)  # Midpoint
    # c = np.array(c)  # Endpoint
    # radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    # angle = np.abs(radians * 180.0/np.pi)
    # angle = np.round(angle, 2)
    # if angle.all() < 0:
    #     angle += 360.0
    # return angle


def classify_pose(landmarks, image, display = False):
    label = "Unknown pose"
    color = (0, 0, 255)  # Red

    time_start = time()

    """Angles for six landmarks"""
    left_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                                        )

    right_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                                         )

    left_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                                         )

    right_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                                        )

    left_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                                        )

    right_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                                        )

    # # Check is both arms are straight
    # if (left_elbow_angle > 165 and left_elbow_angle < 195) and (right_elbow_angle > 165 and right_elbow_angle < 195):
    #     # Check if both shoulders are perpendicular to the body
    #     if (left_shoulder_angle > 80 and left_shoulder_angle < 110) and (right_shoulder_angle > 80 and right_elbow_angle < 110):
    #
    # # Check if Warrior II Pose
    #         # Check if one leg is straight
    #         if (left_knee_angle > 165 and left_knee_angle < 195) or (right_knee_angle > 165 and right_knee_angle < 195):
    #             # Check if other knee is perpendicular
    #             if (left_knee_angle > 80 and left_knee_angle < 110) or (right_knee_angle > 80 and right_knee_angle < 110):
    #                 label = "Warrior II Pose"

    # Note: Camera feed is FLIPPED! Right is left and left is right
    # Check if a shoulder is straight (parallel to the floor, perpendicular to the body)
    if (right_shoulder_angle > 70 and right_shoulder_angle < 110):# or (left_shoulder_angle > 70 and left_shoulder_angle < 110):
        label = "straight arm"

        # Check if elbow is perpendicular
        if (right_elbow_angle > 250 and right_elbow_angle < 290):# or (left_elbow_angle > 70 and left_elbow_angle < 110):
            label = "Hello pose"  # Person is waving, static

    if label is not "Unknown pose":
        color = (0, 255, 0)  # Green

    cv2.putText(image, label, (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 4)
    cv2.putText(image, label, (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    return image, label


def main():
    time1 = 0
    #label = "No pose"
    old_label = None
    record_time_start = time()
    with mp_pose.Pose(min_detection_confidence=0.51, min_tracking_confidence=0.51) as pose:
        while cap.isOpened():
            success, frame = cap.read()

            if not success:  # Sometimes I had to force quit the program, this prevents that. Sort of.
                break

            # Flip the video to be a "mirror" on the laptop. This is less jarring to look at.
            frame = cv2.flip(frame, 1)

            # Convert from BGR to RGB colorspace - MediaPipe likes RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            """Detect the poses"""
            # Make Detections
            results = pose.process(image)

            # Recolor image back to BGR for rendering - OpenCV likes BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Avoid breaking the loop if no frame is extracted
            try:
                landmarks = results.pose_landmarks.landmark
            except:
                # print("No frame extracted")
                pass

            # 1. Right hand
            # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_pose.HAND_CONNECTIONS,
            #                           mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            #                           mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
            #                           )
            #
            # # 2. Left Hand
            # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_pose.HAND_CONNECTIONS,
            #                           mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            #                           mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            #                           )

            # 3. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            time2 = time()
            if (time2 - time1) > 0:
                fps = 1/(time2 - time1)
                # FPS counter in green with a black border to make it easy to read
                cv2.putText(image, 'FPS: {}'.format(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 6)
                cv2.putText(image, 'FPS: {}'.format(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            time1 = time2

            display = False

            # Unpack coordinates and put them in a list
            height, width, depth = image.shape
            landmarks = []
            try:
                if len(landmarks) is 0:
                    for landmark in results.pose_landmarks.landmark:
                        # Append the landmark into the list.
                        landmarks.append((int(landmark.x * width),
                                          int(landmark.y * height),
                                          int((landmark.z * depth)))
                                         )
                    image, label = classify_pose(landmarks, image, display)
            except:
                # No pose detected
                # Needs better error handling
                pass

            # Recording the pose and time it is active.
            if label is not None:
                if label is not old_label:
                    elapsed_time = time()
                    time_record = elapsed_time - record_time_start
                    time_keeping(label, time_record)
                    #if results.landmarks is True:
                    record_time_start = time()

            cv2.imshow('Webcam is running :)', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            old_label = label

        cap.release()
        cv2.destroyAllWindows()


main()
print(ledger)
