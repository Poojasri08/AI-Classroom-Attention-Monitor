import cv2
import mediapipe as mp
import pandas as pd
from datetime import datetime

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False
)

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

NOSE = 1

focused_frames = 0
total_frames = 0
drowsy_frames = 0

status = "Unknown"
head_status = "Unknown"
attention_percentage = 0

data = []

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    h, w, c = frame.shape

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            eye_points = []

            for id in LEFT_EYE + RIGHT_EYE:

                landmark = face_landmarks.landmark[id]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                eye_points.append((x, y))

            top = eye_points[1][1]
            bottom = eye_points[5][1]

            eye_open = abs(top - bottom)

            if eye_open < 5:
                drowsy_frames += 1
            else:
                drowsy_frames = 0

            if drowsy_frames > 20:
                status = "Drowsy"
            else:
                status = "Attentive"

            nose = face_landmarks.landmark[NOSE]

            nose_x = int(nose.x * w)

            cv2.circle(frame, (nose_x, 100), 5, (255, 0, 0), -1)

            if nose_x < w // 3:
                head_status = "Looking Left"

            elif nose_x > 2 * (w // 3):
                head_status = "Looking Right"

            else:
                head_status = "Focused"

            total_frames += 1

            if head_status == "Focused" and status == "Attentive":
                focused_frames += 1

            attention_percentage = int(
                (focused_frames / total_frames) * 100
            )

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data.append({
            "timestamp": datetime.now(),
            "status": status,
               "head_status": head_status,
                 "attention_percentage": attention_percentage
                 })

    cv2.putText(frame,
                f"Status: {status}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.putText(frame,
                f"Head: {head_status}",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2)

    cv2.putText(frame,
                f"Attention: {attention_percentage}%",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2)

    cv2.imshow("Classroom Attention AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



df=pd.DataFrame(data)
df.to_csv("attention_data.csv", index=False)    
print("Data saved to attention_data.csv")
cap.release()
cv2.destroyAllWindows()