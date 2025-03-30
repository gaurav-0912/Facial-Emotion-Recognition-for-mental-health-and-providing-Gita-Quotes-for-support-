import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
from keras.models import load_model
import random
import csv

warnings.filterwarnings("ignore")

# Load the emotion detection model
model = load_model("C:\\Users\\gaura\\OneDrive\\Desktop\\major\\best_model.keras")
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to load quotes
def load_quotes(file_path):
    emotion_categories = {
        "optimistic": [],
        "thankful": [],
        "empathetic": [],
        "pessimistic": [],
        "anxious": [],
        "sad": [],
        "annoyed": [],
        "denial": [],
        "official report": [],
        "surprise": [],
        "joking": []
    }
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            quote = row[1].strip()
            if float(row[2]) == 1.0:
                emotion_categories["optimistic"].append(quote)
            if float(row[3]) == 1.0:
                emotion_categories["thankful"].append(quote)
            if float(row[4]) == 1.0:
                emotion_categories["empathetic"].append(quote)
            if float(row[5]) == 1.0:
                emotion_categories["pessimistic"].append(quote)
            if float(row[6]) == 1.0:
                emotion_categories["anxious"].append(quote)
            if float(row[7]) == 1.0:
                emotion_categories["sad"].append(quote)
            if float(row[8]) == 1.0:
                emotion_categories["annoyed"].append(quote)
            if float(row[9]) == 1.0:
                emotion_categories["denial"].append(quote)
            if float(row[10]) == 1.0:
                emotion_categories["official report"].append(quote)
            if float(row[11]) == 1.0:
                emotion_categories["surprise"].append(quote)
            if float(row[12]) == 1.0:
                emotion_categories["joking"].append(quote)
    return emotion_categories

# Function to fetch quotes based on sentiment
def fetch_quote_based_on_sentiment(sentiment, emotion_categories):
    sentiment_to_emotion = {
        "positive": ["optimistic", "thankful", "joking"],
        "negative": ["pessimistic", "sad", "anxious", "annoyed"],
        "neutral": ["empathetic", "official report", "surprise"]
    }
    possible_emotions = sentiment_to_emotion.get(sentiment, [])
    quotes = [quote for emotion in possible_emotions for quote in emotion_categories.get(emotion, [])]
    if quotes:
        return random.choice(quotes)
    return "No quote available for this sentiment."

# Path to the file containing quotes
file_path = r"C:\Users\gaura\OneDrive\Desktop\major\chapter 1--.txt"



emotion_categories = load_quotes(file_path)

# Start video capture for emotion detection
cap = cv2.VideoCapture(0)
emotion_detected = False  # Flag to check if an emotion has been detected

while not emotion_detected:
    ret, test_img = cap.read()
    if not ret:
        continue

    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        # Map detected emotion to sentiment
        sentiment = "neutral"
        if predicted_emotion in ['happy', 'surprise']:
            sentiment = "positive"
        elif predicted_emotion in ['angry', 'disgust', 'sad', 'fear']:
            sentiment = "negative"

        cv2.putText(test_img, f"{predicted_emotion} ({sentiment})", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Set the flag to True after detecting the first emotion
        emotion_detected = True
        break

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial Emotion Analysis', resized_img)

    if cv2.waitKey(10) == ord('q') or emotion_detected:
        break

cap.release()
cv2.destroyAllWindows()

# Ask the user for the reason behind their emotion and provide a quote
user_reason = input(f"Detected Emotion: {predicted_emotion}. Please share the reason for your emotion: ")
quote = fetch_quote_based_on_sentiment(sentiment, emotion_categories)
print(f"Here’s a quote for you: {quote}")

