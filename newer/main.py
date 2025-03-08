import cv2
import os
import numpy as np


# Load YOLO
cfg_path = r"D:\AI-based-Traffic-Control-System--\\newer\\yolo\\yolov3.cfg"
weights_path = r"D:\AI-based-Traffic-Control-System--\\newer\\yolo\\yolov3.weights"
coco_names_path = r"D:\AI-based-Traffic-Control-System--\\newer\\coco.names"
video_path = r"D:\AI-based-Traffic-Control-System--\datas\\video3.mp4"
#Ensure required files exist
if not all(os.path.exists(path) for path in [cfg_path, weights_path, coco_names_path, video_path]):
    raise FileNotFoundError("One or more required files are missing!")

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
# GPU can be accessed
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO labels
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Filter only vehicles
vehicle_classes = {"car", "bus", "truck", "motorbike", "bicycle"}

# Open video
cap = cv2.VideoCapture(video_path)

# Get original video properties
frame_width = int(cap.get(3))  # Width
frame_height = int(cap.get(4))  # Height
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second

frame_skip = 2  # Skip every 1 frame for faster processing
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End if video is finished

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frames to speed up

    height, width, _ = frame.shape

    # Prepare YOLO input **without cropping**
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    class_ids = []
    confidences = []
    boxes = []

    for out_data in outs:  # 🔹 Rename loop variable to prevent overwriting 'out'
        for detection in out_data:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter only vehicles
            if confidence > 0.5 and classes[class_id] in vehicle_classes:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box


    # Show video
    cv2.imshow("Vehicle Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(10) & 0xFF == ord("q"):  # Increase wait time for faster playback
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

