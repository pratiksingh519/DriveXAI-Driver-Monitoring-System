import threading
import playsound

ALARM_ON = False

def play_alarm(path):
    global ALARM_ON
    if not ALARM_ON:
        ALARM_ON = True
        threading.Thread(target=playsound.playsound,
                         args=(path,),
                         daemon=True).start()

def stop_alarm():
    global ALARM_ON
    ALARM_ON = False