import cv2
import mediapipe as mp
import PySimpleGUI as sg

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Create a list of available webcams
num_cameras = 3  # Change this to the number of available webcams
cameras = [cv2.VideoCapture(i) for i in range(num_cameras)]

# Create a GUI for selecting the webcam
layout = [[sg.Text("Select a camera:")]]
for i in range(num_cameras):
    layout.append([sg.Radio(f"Camera {i}", group_id="camera", default=i==0)])

layout.append([sg.Button("OK")])
window = sg.Window("Select Camera", layout)

event, values = window.read()
if event == "OK":
    camera_index = [i for i in range(num_cameras) if values[i]][0]
else:
    exit()

# Start capturing frames from the selected webcam
cap = cameras[camera_index]

# Initialize Mediapipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        # Flip the image horizontally for a mirror effect
        image = cv2.flip(image, 1)

        # Convert the image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Set flags to allow drawing of the landmarks and connections on the image
        mp_drawing_styles = mp.solutions.drawing_styles
        drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style()
        drawing_spec2 = mp_drawing_styles.get_default_pose_connection_style()

        # Process the image with Mediapipe Pose
        results = pose.process(image)

        # Get the coordinates of the right shoulder, right hip, and nose landmarks
        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            nose = landmarks[mp_pose.PoseLandmark.NOSE]

            # Calculate the angle between the right arm and upper body
            if right_shoulder.visibility > 0 and right_hip.visibility > 0 and nose.visibility > 0:
                dx = right_hip.x - right_shoulder.x
                dy = right_hip.y - right_shoulder.y
                angle = int(round(180 - abs(math.degrees(math.atan2(dy, dx))), 0))

                # Draw the angle on the image
                cv2.putText(image, f"Angle: {angle} degrees", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Convert the image back to BGR for display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Display the image
        cv2.imshow("Pose Estimation", image)

        # Exit the program if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy the window
    cap.release()
    cv2.destroyAllWindows()
