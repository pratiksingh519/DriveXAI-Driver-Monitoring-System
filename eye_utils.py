import numpy as np

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