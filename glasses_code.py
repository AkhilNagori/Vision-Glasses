import RPi.GPIO as GPIO
import torch
import cv2
import numpy as np
import subprocess
import time
from model.crnn import CRNN
from utils import preprocess_image, decode_predictions
import espeakng

BUTTON_PIN = 17

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(imgH=32, nc=1, nclass=80, nh=256)
model.load_state_dict(torch.load("model/crnn_trained.pth", map_location=device))
model.eval().to(device)

def capture_and_process():
    image_path = "savimg.jpg"
    subprocess.run(["libcamera-jpeg", "-o", image_path, "--width", "640", "--height", "480"])
    image = cv2.imread(image_path)
    if image is None:
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = preprocess_image(gray)
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(img)
        decoded_text = decode_predictions(preds)
    if decoded_text:
        tts = espeakng.Speaker()
        tts.wpm = 100
        tts.say(decoded_text.replace("\r\n", " "), wait4prev=True)

try:
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.HIGH:
            capture_and_process()
            time.sleep(1)
except KeyboardInterrupt:
    GPIO.cleanup()
# Cleanup GPIO on exit
finally:
    GPIO.cleanup()
    print("GPIO cleaned up and program exited.")
# This code captures an image when a button is pressed, processes it using a CRNN model,
# and uses text-to-speech to read the recognized text aloud. It uses the RPi.GPIO library for button input,
# OpenCV for image processing, and a custom CRNN model for text recognition.
# The espeakng library is used for text-to-speech functionality.
# Ensure you have the necessary libraries installed:
# pip install torch torchvision opencv-python espeakng RPi.GPIO
# Ensure you have the CRNN model and utils module available in your project.
# The model should be trained and saved as 'crnn_trained.pth' in the 'model' directory.