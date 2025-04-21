from deepface import DeepFace
import cv2

# Path to your image file
img_path = r"D:\clg\DL\face_facial_hair_fine_looking_guy_man_model_person_portrait-1563283.jpg!d"

# Perform emotion analysis using DeepFace
result = DeepFace.analyze(img_path=img_path, actions=["emotion"])

# Extract the predicted emotion
predicted_emotion = result[0]['dominant_emotion']

# Load the image to display the result on
img = cv2.imread(img_path)

# Set the text and font for displaying the emotion
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, f'Emotion: {predicted_emotion}', (50, 50), font, 1, (0, 255, 0))

# Show the image with the predicted emotion
cv2.imshow('Emotion Detection', img)

# Wait for a key press and close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the result in a readable format
print(f"Predicted Emotion: {predicted_emotion}")
print(result)
