import cv2
from cvzone.HandTrackingModule import HandDetector
import tensorflow as tf
import numpy as np
import math

# Load labels from file
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Setup webcam and hand detection
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Crop hand with padding and center on white square
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgHeight, imgWidth = img.shape[:2]
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(x + w + offset, imgWidth)
        y2 = min(y + h + offset, imgHeight)
        imgCrop = img[y1:y2, x1:x2]
        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + min(wCal, imgSize)] = imgResize[:, :min(wCal, imgSize)]
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + min(hCal, imgSize), :] = imgResize[:min(hCal, imgSize), :]

            # Preprocess for TFLite model (assumes 224x224 input)
            input_img = cv2.resize(imgWhite, (224, 224))
            input_img = np.expand_dims(input_img.astype(np.float32) / 255, axis=0)

            interpreter.set_tensor(input_details[0]['index'], input_img)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]
            index = int(np.argmax(prediction))
            confidence = float(prediction[index])

            # Display label and confidence
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 140, y - offset - 5), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, f"{labels[index]} ({confidence:.2f})", (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        except Exception as e:
            print("Classification error:", e)

        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 3)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("ASL Detection", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
