import cv2

def detect_head_direction(frame, nose_x, face_center):
    if nose_x < face_center - 40:
        cv2.putText(frame, "LOOKING LEFT", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    elif nose_x > face_center + 40:
        cv2.putText(frame, "LOOKING RIGHT", (350, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:
        cv2.putText(frame, "FOCUSED", (250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)