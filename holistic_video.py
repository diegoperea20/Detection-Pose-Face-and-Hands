import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture("video.mp4")

with mp_holistic.Holistic(
     static_image_mode=False,
     model_complexity=1) as holistic:

     cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)  # Resizable window

     while True:
          ret, frame = cap.read()
          if not ret:
               break

          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          results = holistic.process(frame_rgb)

          # Face
          mp_drawing.draw_landmarks(
               frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
               mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
               mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2))
          
          # Left hand (blue)
          mp_drawing.draw_landmarks(
               frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
               mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=1),
               mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))
          
          # Right hand (green)
          mp_drawing.draw_landmarks(
               frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
               mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
               mp_drawing.DrawingSpec(color=(57, 143, 0), thickness=2))
          
          # Pose
          mp_drawing.draw_landmarks(
               frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
               mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=2, circle_radius=1),
               mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

          frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Adjust frame size
          frame = cv2.flip(frame, 1)
          cv2.imshow("Frame", frame)

          if cv2.waitKey(1) & 0xFF == 27:
               break

cap.release()
cv2.destroyAllWindows()
