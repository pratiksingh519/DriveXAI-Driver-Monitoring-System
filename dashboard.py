import cv2
import mediapipe as mp
import numpy as np
import winsound
import time
import pyttsx3
import threading
import csv

# -------- SETTINGS --------
EYE_AR_THRESH = 0.30
EYE_CLOSE_TIME = 4
YAWN_TIME = 2

# -------- STATE --------
eye_closed_start = None
yawn_start = None

last_beep_time = 0

drowsy_count = 0
yawn_count = 0
left_count = 0
right_count = 0

event_history = []

# -------- CSV --------
log_file = open("driver_log.csv", "a", newline='')
logger = csv.writer(log_file)

# -------- VOICE --------
engine = pyttsx3.init()

def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

# -------- MEDIAPIPE --------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# -------- FUNCTIONS --------
def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(np.array(mouth[2]) - np.array(mouth[10]))
    B = np.linalg.norm(np.array(mouth[4]) - np.array(mouth[8]))
    C = np.linalg.norm(np.array(mouth[0]) - np.array(mouth[6]))
    return (A + B) / (2.0 * C)

def beep():
    global last_beep_time
    if time.time() - last_beep_time > 1:
        winsound.Beep(1000, 500)
        last_beep_time = time.time()

# -------- CAMERA --------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # -------- LOW LIGHT FIX --------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.equalizeHist(gray)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # -------- EYES --------
            left_idx = [33,160,158,133,153,144]
            right_idx = [362,385,387,263,373,380]

            left_eye = [(int(face_landmarks.landmark[i].x*w),
                         int(face_landmarks.landmark[i].y*h)) for i in left_idx]

            right_eye = [(int(face_landmarks.landmark[i].x*w),
                          int(face_landmarks.landmark[i].y*h)) for i in right_idx]

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

            cv2.putText(frame, f"EAR:{round(ear,2)}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # -------- DROWSINESS --------
            if ear < EYE_AR_THRESH:

                if eye_closed_start is None:
                    eye_closed_start = time.time()

                elapsed = time.time() - eye_closed_start

                cv2.putText(frame, f"Closed:{round(elapsed,1)}s", (200,150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                if elapsed >= EYE_CLOSE_TIME:
                    cv2.putText(frame, "DROWSINESS ALERT", (100,60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

                    beep()
                    speak("Wake up driver")

                    drowsy_count += 1
                    logger.writerow(["DROWSY", time.strftime('%H:%M:%S')])
                    event_history.append("Drowsy " + time.strftime('%H:%M:%S'))

                    # 🔥 RESET (IMPORTANT FIX)
                    eye_closed_start = None

            else:
                eye_closed_start = None

            # -------- YAWNING --------
            mouth_idx = [61,291,81,178,13,14,402,318,311,308,415,310]
            mouth = [(int(face_landmarks.landmark[i].x*w),
                      int(face_landmarks.landmark[i].y*h)) for i in mouth_idx]

            mar = mouth_aspect_ratio(mouth)

            if mar > 0.6:
                if yawn_start is None:
                    yawn_start = time.time()

                if time.time() - yawn_start >= YAWN_TIME:
                    cv2.putText(frame, "YAWNING", (200,100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

                    yawn_count += 1
                    logger.writerow(["YAWN", time.strftime('%H:%M:%S')])
                    event_history.append("Yawn " + time.strftime('%H:%M:%S'))

                    yawn_start = None
            else:
                yawn_start = None

    # -------- COUNTERS --------
    cv2.putText(frame,f"Drowsy:{drowsy_count}",(10,300),0,0.6,(0,0,255),2)
    cv2.putText(frame,f"Yawns:{yawn_count}",(10,330),0,0.6,(0,255,255),2)

    # -------- HISTORY --------
    y = 50
    for ev in event_history[-5:]:
        cv2.putText(frame, ev, (350,y),0,0.5,(255,255,255),1)
        y += 20

    cv2.imshow("DriveXAI FINAL", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()