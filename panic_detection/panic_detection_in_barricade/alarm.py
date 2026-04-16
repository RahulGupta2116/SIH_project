import subprocess

# Global process handle (so only one alarm plays at a time)
current_alarm = None

def _play(file):
    global current_alarm
    # Stop previous alarm
    stop_alarm()
    # Start new alarm (non-blocking, extremely lightweight)
    current_alarm = subprocess.Popen(
        ["aplay", "-q", file],  # -q = quiet (no console spam)
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def stop_alarm():
    global current_alarm
    if current_alarm and current_alarm.poll() is None:
        current_alarm.terminate()
    current_alarm = None

def beep():
    _play("beep.wav")  # Warning beep

def fast_beep():
    _play("alarm.wav")  # Same file, can change later if needed

def siren():
    _play("siren.wav")    # Panic siren

