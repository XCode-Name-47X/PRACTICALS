import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import matplotlib.pyplot as plt
image_path = r"D:\clg\DL\face_facial_hair_fine_looking_guy_man_model_person_portrait-1563283.jpg!d"
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, minNeighbors=5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
img_rgb_overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb_overlay)
plt.axis("off")
plt.show()
print("Detected Faces:", faces)
