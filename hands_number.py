import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

def palm_centroid(coordinates_list):
    coordinates = np.array(coordinates_list)
    centroid = np.mean(coordinates, axis=0)
    centroid = int(centroid[0]), int(centroid[1])
    return centroid

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Thumb  (Pulgar)
thumb_points = [1, 2, 4]

#  Index finger, middle finger, ring finger and little finger  (Índice, medio, anular y meñique)
palm_points = [0, 1, 2, 5, 9, 13, 17]
fingertips_points = [8, 12, 16, 20]
finger_base_points = [6, 10, 14, 18]

# Colors (Colores)
GREEN = (48, 255, 48)
BLUE = (192, 101, 21)
YELLOW = (0, 204, 255)
PURPLE = (128, 64, 128)
PEACH = (180, 229, 255)

with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coordinates_thumb = []
                coordinates_palm = []
                coordinates_ft = []
                coordinates_fb = []
                for index in thumb_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_thumb.append([x, y])

                for index in palm_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_palm.append([x, y])

                for index in fingertips_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_ft.append([x, y])

                for index in finger_base_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_fb.append([x, y])

                # Rest of the code for each hand (calculations and visualization) (Resto del código para cada mano (cálculos y visualización))
                ##########################
                # Thumb (Pulgar)
                p1 = np.array(coordinates_thumb[0])
                p2 = np.array(coordinates_thumb[1])
                p3 = np.array(coordinates_thumb[2])

                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)

                # Calculate the angle (Calcular el ángulo)
                angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                thumb_finger = np.array(False)
                if angle > 150:
                    thumb_finger = np.array(True)

                ################################
                #  Index finger, middle finger, ring finger and little finger (Índice, medio, anular y meñique)
                nx, ny = palm_centroid(coordinates_palm)
                cv2.circle(frame, (nx, ny), 3, (0, 255, 0), 2)
                coordinates_centroid = np.array([nx, ny])
                coordinates_ft = np.array(coordinates_ft)
                coordinates_fb = np.array(coordinates_fb)

                # Distances (Distancias)
                d_centroid_ft = np.linalg.norm(coordinates_centroid - coordinates_ft, axis=1)
                d_centroid_fb = np.linalg.norm(coordinates_centroid - coordinates_fb, axis=1)
                dif = d_centroid_ft - d_centroid_fb
                fingers = dif > 0
                fingers = np.append(thumb_finger, fingers)
                fingers_counter = str(np.count_nonzero(fingers == True))

                thickness = [2, 2, 2, 2, 2]

                for (i, finger) in enumerate(fingers):
                    if finger == True:
                        thickness[i] = -1

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                ################################
                #Visualization  Visualización
               
               
                #Right rectangle  (Rectángulo derecho )
                cv2.rectangle(frame, (width - 80, 0), (width, 80), (0, 0, 255), -1)
                cv2.putText(frame, fingers_counter, (width - 65, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Thumb (Pulgar)
                thumb_tip = np.array([int(hand_landmarks.landmark[4].x * width), int(hand_landmarks.landmark[4].y * height)])
                cv2.rectangle(frame, (thumb_tip[0] - 20, thumb_tip[1] - 20), (thumb_tip[0] + 20, thumb_tip[1] + 20), PEACH, thickness[0])
                cv2.putText(frame, "Thumb", (thumb_tip[0] - 35, thumb_tip[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Index (Índice)
                index_tip = np.array([int(hand_landmarks.landmark[8].x * width), int(hand_landmarks.landmark[8].y * height)])
                cv2.rectangle(frame, (index_tip[0] - 20, index_tip[1] - 20), (index_tip[0] + 20, index_tip[1] + 20), PURPLE, thickness[1])
                cv2.putText(frame, "Index", (index_tip[0] - 30, index_tip[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Middle (Medio)
                middle_tip = np.array([int(hand_landmarks.landmark[12].x * width), int(hand_landmarks.landmark[12].y * height)])
                cv2.rectangle(frame, (middle_tip[0] - 20, middle_tip[1] - 20), (middle_tip[0] + 20, middle_tip[1] + 20), YELLOW, thickness[2])
                cv2.putText(frame, "Middle", (middle_tip[0] - 30, middle_tip[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # ring finger (Anular)
                ring_tip = np.array([int(hand_landmarks.landmark[16].x * width), int(hand_landmarks.landmark[16].y * height)])
                cv2.rectangle(frame, (ring_tip[0] - 20, ring_tip[1] - 20), (ring_tip[0] + 20, ring_tip[1] + 20), GREEN, thickness[3])
                cv2.putText(frame, "Ring", (ring_tip[0] - 30, ring_tip[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Little Finger (Menique9
                pinky_tip = np.array([int(hand_landmarks.landmark[20].x * width), int(hand_landmarks.landmark[20].y * height)])
                cv2.rectangle(frame, (pinky_tip[0] - 20, pinky_tip[1] - 20), (pinky_tip[0] + 20, pinky_tip[1] + 20), BLUE, thickness[4])
                cv2.putText(frame, "Litle", (pinky_tip[0] - 30, pinky_tip[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
