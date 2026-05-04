# 🚗 DriveXAI - Driver Monitoring System

An AI-powered Driver Monitoring System built using Computer Vision to enhance road safety by detecting driver fatigue and unsafe behavior in real-time.

---

## 📌 Overview

DriveXAI is a real-time driver safety system that monitors eye movements, facial behavior, and head position using AI.  
It helps prevent accidents by detecting drowsiness, yawning, and distraction, and immediately alerts the driver.

---

## 🔥 Key Features

- Drowsiness Detection (eye closure tracking)
- Yawning Detection
- Real-Time Alerts (Beep + Voice)
- Event Logging (CSV)
- Fatigue Monitoring
- Works in Low-Light Conditions

---

## 🛠️ Tech Stack

- Python  
- OpenCV  
- MediaPipe  
- NumPy  
- pyttsx3  

---

## ⚙️ How It Works

- Webcam captures real-time video  
- Face landmarks detected using MediaPipe  
- Eye Aspect Ratio (EAR) calculated  
- If eyes closed > 4 seconds:
  - Alert triggered  
  - Beep sound  
  - Voice: "Wake up driver"

---

## 📂 Project Structure

DriveXAI/
│── main.py
│── dashboard.py
│── config.py
│── eye_utils.py
│── detection_utils.py
│── alarm.py
│── eve_engine.py
│── driver_log.csv
│── README.md

---

## ▶️ Run Project

pip install mediapipe opencv-python numpy pyttsx3  
python main.py  

---

## 📊 Output

- Real-time camera feed  
- Alerts on screen  
- CSV logging  

---

## 🚀 Future Improvements

- YOLO phone detection  
- Dashboard UI  
- Mobile integration  

---

## 👨‍💻 Author

Pratik Singh  
