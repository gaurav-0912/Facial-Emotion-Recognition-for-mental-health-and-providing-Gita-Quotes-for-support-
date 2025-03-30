import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model
from textblob import TextBlob
import random

# Load the emotion detection model
model = load_model("C:/Users/gaura/OneDrive/Desktop/major/best_model.keras")
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"

def fetch_quote_based_on_sentiment(sentiment):
    quotes = {
        "positive": [],
        "negative": [],
        "neutral": []
    }

    with open('chapter 1--.txt', 'r') as file:
        for line in file:
            if "some positive keyword" in line:
                quotes["positive"].append(line.strip())
            elif "some negative keyword" in line:
                quotes["negative"].append(line.strip())
            else:
                quotes["neutral"].append(line.strip())

    if quotes[sentiment]:
        return random.choice(quotes[sentiment])
    else:
        return "No quote available for this sentiment."

while True:
    ret, test_img = cap.read()
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Ask user for the reason of emotion
        reason = input("Please enter the reason for your emotion: ")
        sentiment = analyze_sentiment(reason)
        quote = fetch_quote_based_on_sentiment(sentiment)
        print(f"Quote based on your emotion: {quote}")

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()