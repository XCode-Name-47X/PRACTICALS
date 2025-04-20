import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
image_path = r"D:\clg\DL\face_facial_hair_fine_looking_guy_man_model_person_portrait-1563283.jpg!d"
image = Image.open(image_path).convert("RGB")
image_tensor = F.to_tensor(image).unsqueeze(0)
with torch.no_grad():
    predictions = model(image_tensor)
boxes = predictions[0]["boxes"]
labels = predictions[0]["labels"]
scores = predictions[0]["scores"]
threshold = 0.8  
human_boxes = [box for i, box in enumerate(boxes) if labels[i] == 1 and scores[i] > threshold]
fig, ax = plt.subplots(1, figsize=(10, 6))
ax.imshow(image)
for box in human_boxes:
    x1, y1, x2, y2 = box
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none")
    ax.add_patch(rect)
plt.show()
