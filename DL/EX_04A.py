import cv2
import numpy as np

config = r'D:\clg\DL\yolo\yolov3.cfg'
weights = r'D:\clg\DL\yolo\yolov3.weights'
names = r'D:\clg\DL\yolo\coco.names'
with open(names, 'r') as f:
    classes = f.read().strip().split('\n')
net = cv2.dnn.readNet(weights, config)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
cap = cv2.VideoCapture(r'D:\clg\DL\yolo\2053100-uhd_3840_2160_30fps.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    detected = set()
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                label = classes[class_id]
                detected.add(label)
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if detected:
        print("Detected:", ', '.join(detected))
    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
