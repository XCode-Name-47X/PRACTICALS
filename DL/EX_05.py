import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace

# Suppress OpenMP error in some environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load image
image_path = r"D:\clg\DL\face_facial_hair_fine_looking_guy_man_model_person_portrait-1563283.jpg!d"
img = cv2.imread(image_path)

# Check if the image was loaded correctly
if img is None:
    raise ValueError(f"Error loading image at path: {image_path}")

# Convert BGR (OpenCV format) to RGB (Matplotlib format)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Perform emotion analysis
try:
    result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
    dominant_emotion = result[0]['dominant_emotion']
except Exception as e:
    raise ValueError(f"Error in DeepFace analysis: {e}")

# Basic Sentiment Mapping (instead of incorrect LSTM usage)
emotion_to_sentiment = {
    "happy": "Positive",
    "surprise": "Positive",
    "neutral": "Neutral",
    "sad": "Negative",
    "angry": "Negative",
    "fear": "Negative",
    "disgust": "Negative"
}
final_sentiment = emotion_to_sentiment.get(dominant_emotion, "Neutral")

# Overlay text on the image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, f"Emotion: {dominant_emotion}", (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(img, f"Sentiment: {final_sentiment}", (10, 90), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

# Convert back to RGB for displaying
img_rgb_overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the result image
plt.imshow(img_rgb_overlay)
plt.title(f"Emotion: {dominant_emotion}, Sentiment: {final_sentiment}")
plt.axis("off")
plt.show()

# Print detected emotions and sentiment
print("Emotion Analysis Result:", result)
print("Predicted Sentiment:", final_sentiment)
