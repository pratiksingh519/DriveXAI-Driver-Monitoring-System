import cv2
import mediapipe as mp
import numpy as np
import winsound
import time
import csv
import pyttsx3

# ---------------- SETTINGS ----------------
USE_YOLO = False  # True करोगे तो phone detection ON (ultralytics install होना चाहिए)

EYE_AR_THRESH = 0.25
MOUTH_THRESH = 0.6
EYE_CLOSE_TIME = 4      # sec
YAWN_TIME = 2           # sec

# ---------------- STATE ----------------
eye_closed_start = None
yawn_start = None

blink_count = 0
blink_temp = 0

last_beep_time = 0
last_voice_time = 0

fatigue_score = 0

# Counters
drowsy_count = 0
yawn_count = 0
left_count = 0
right_count = 0

# History
event_history = []

# CSV logger
log_file = open("driver_log.csv", "a", newline='')
logger = csv.writer(log_file)

# Voice
engine = pyttsx3.init()
def speak(text):
    global last_voice_time
    now = time.time()
    if now - last_voice_time > 8:
        engine.say(text)
        engine.runAndWait()
        last_voice_time = now

# Optional YOLO
if USE_YOLO:
    from ultralytics import YOLO
    yolo_model = YOLO("yolov8n.pt")

# Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# ---------------- HELPERS ----------------
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
    now = time.time()
    if now - last_beep_time > 1:
        winsound.Beep(1000, 400)
        last_beep_time = now

# Adaptive threshold
ear_values = []
calibrated = False

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]

    # YOLO phone detection
    if USE_YOLO:
        results = yolo_model(frame)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = yolo_model.names[cls]
                if label == "cell phone":
                    cv2.putText(frame, "PHONE DETECTED", (200,150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                    logger.writerow(["PHONE", time.strftime('%H:%M:%S')])

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Red face overlay
            face_points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
            hull = cv2.convexHull(np.array(face_points))
            overlay = frame.copy()
            cv2.fillPoly(overlay, [hull], (0, 0, 255))
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

            # Eyes
            left_idx = [33,160,158,133,153,144]
            right_idx = [362,385,387,263,373,380]

            left_eye = [(int(face_landmarks.landmark[i].x*w),
                         int(face_landmarks.landmark[i].y*h)) for i in left_idx]
            right_eye = [(int(face_landmarks.landmark[i].x*w),
                          int(face_landmarks.landmark[i].y*h)) for i in right_idx]

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

            # Calibration
            if not calibrated:
                ear_values.append(ear)
                if len(ear_values) > 100:
                    avg_ear = sum(ear_values)/len(ear_values)
                    EYE_AR_THRESH = avg_ear * 0.75
                    calibrated = True

            # Mouth
            mouth_idx = [61,291,81,178,13,14,402,318,311,308,415,310]
            mouth = [(int(face_landmarks.landmark[i].x*w),
                      int(face_landmarks.landmark[i].y*h)) for i in mouth_idx]
            mar = mouth_aspect_ratio(mouth)

            # Blink
            if ear < EYE_AR_THRESH:
                blink_temp += 1
            else:
                if blink_temp > 2:
                    blink_count += 1
                blink_temp = 0

            # Drowsiness
            if ear < EYE_AR_THRESH:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                if time.time() - eye_closed_start >= EYE_CLOSE_TIME:
                    cv2.putText(frame, "DROWSY", (120,50), 0,1,(0,0,255),3)
                    beep()
                    speak("Wake up driver")
                    drowsy_count += 1
                    fatigue_score += 2
                    logger.writerow(["DROWSY", time.strftime('%H:%M:%S')])
                    event_history.append("Drowsy " + time.strftime('%H:%M:%S'))
                    eye_closed_start = None
            else:
                eye_closed_start = None

            # Yawning
            if mar > MOUTH_THRESH:
                if yawn_start is None:
                    yawn_start = time.time()
                if time.time() - yawn_start >= YAWN_TIME:
                    cv2.putText(frame, "YAWNING", (200,100), 0,1,(0,0,255),3)
                    yawn_count += 1
                    fatigue_score += 1
                    logger.writerow(["YAWN", time.strftime('%H:%M:%S')])
                    event_history.append("Yawn " + time.strftime('%H:%M:%S'))
                    yawn_start = None
            else:
                yawn_start = None

            # Head direction
            nose = face_landmarks.landmark[1]
            nose_x = int(nose.x*w)
            center = (left_eye[0][0] + right_eye[3][0]) // 2

            if nose_x < center - 50:
                cv2.putText(frame,"LEFT",(10,30),0,1,(255,0,0),2)
                left_count += 1
                logger.writerow(["LEFT", time.strftime('%H:%M:%S')])
            elif nose_x > center + 50:
                cv2.putText(frame,"RIGHT",(350,30),0,1,(255,0,0),2)
                right_count += 1
                logger.writerow(["RIGHT", time.strftime('%H:%M:%S')])
            else:
                cv2.putText(frame,"FOCUSED",(250,30),0,1,(0,255,0),2)

    # Status
    if fatigue_score > 15:
        status = "CRITICAL"
        color = (0,0,255)
    elif fatigue_score > 5:
        status = "WARNING"
        color = (0,255,255)
    else:
        status = "SAFE"
        color = (0,255,0)

    cv2.putText(frame, f"Status: {status}", (10,450), 0,0.7,color,2)

    # Counters
    cv2.putText(frame, f"Drowsy:{drowsy_count}", (10,300),0,0.6,(0,0,255),2)
    cv2.putText(frame, f"Yawns:{yawn_count}", (10,330),0,0.6,(0,255,255),2)
    cv2.putText(frame, f"Left:{left_count}", (10,360),0,0.6,(255,0,0),2)
    cv2.putText(frame, f"Right:{right_count}", (10,390),0,0.6,(255,0,0),2)
    cv2.putText(frame, f"Blinks:{blink_count}", (10,420),0,0.6,(200,200,200),2)

    # Last 5 events
    y_offset = 50
    for ev in event_history[-5:]:
        cv2.putText(frame, ev, (350,y_offset), 0,0.5,(255,255,255),1)
        y_offset += 20

    cv2.imshow("DriveXAI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()