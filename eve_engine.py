import numpy as np

class EVEEngine:
    def __init__(self, thresh=0.25, frames=15):
        self.thresh = thresh
        self.frames = frames
        self.counter = 0

    def eye_aspect_ratio(self, eye):
        A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        return (A + B) / (2.0 * C)

    def check(self, leftEye, rightEye):
        ear = (self.eye_aspect_ratio(leftEye) +
               self.eye_aspect_ratio(rightEye)) / 2.0

        if ear < self.thresh:
            self.counter += 1
            if self.counter >= self.frames:
                return "DROWSY"
        else:
            self.counter = 0

        return "AWAKE"