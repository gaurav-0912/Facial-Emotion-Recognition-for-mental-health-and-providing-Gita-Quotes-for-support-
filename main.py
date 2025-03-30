import cv2
from emotion_detection import detect_emotion, emotion_model
from textblob import TextBlob
import random
import csv

# Load Haar Cascade for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"

def load_quotes(file_path):
    emotion_categories = {
        "optimistic": [], "thankful": [], "empathetic": [],
        "pessimistic": [], "anxious": [], "sad": [], "annoyed": [],
        "denial": [], "official report": [], "surprise": [], "joking": []
    }
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            quote = row[1].strip()
            if float(row[2]) == 1.0:
                emotion_categories["optimistic"].append(quote)
            # Repeat for other emotions as per CSV structure
    return emotion_categories

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

# Load quotes from text file
file_path = "C:/Users/gaura/OneDrive/Desktop/major/chapter 1.txt"
quotes = load_quotes(file_path)

# Main loop for capturing and analyzing video feed
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Detect emotion in the face region
        face_region = gray_frame[y:y + h, x:x + w]
        emotion = detect_emotion(face_region, emotion_model)

        # Display the detected emotion on the frame
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Prompt for user input if an emotion is detected
        if emotion:
            reason = input(f"Detected emotion: {emotion}. Please enter the reason for your emotion: ")
            sentiment = analyze_sentiment(reason)  # Analyze sentiment from user input
            quote = fetch_quote_based_on_sentiment(sentiment, quotes)  # Get a quote based on sentiment
            print(f"Here’s a quote for you: {quote}")

    # Display the output
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(10) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
