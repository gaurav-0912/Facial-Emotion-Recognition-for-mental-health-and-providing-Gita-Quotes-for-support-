import csv
import random

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
# Specify the path to your text file
file_path = "C:/Users/gaura/OneDrive/Desktop/major/chapter 1.txt"  # Use forward slashes or escape backslashes
quotes = load_quotes(file_path)
print(quotes)  # Print or process the loaded quotes as needed
