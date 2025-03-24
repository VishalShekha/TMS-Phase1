import numpy as np
import cv2
import pathlib
import time

class BoundedBox:
    def __init__(self, xmin, ymin, xmax, ymax, ids, confidence):
        with open(str(pathlib.Path.cwd().parents[0]) + "/datas/coco.names", 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')  # stores a list of classes
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax       
        self.ymax = ymax
        self.name = self.classes[ids]
        self.confidence = confidence

class Lane:
    def __init__(self, count, frame, lane_number):
        self.count = count  # Number of vehicles in the lane
        self.frame = frame  # Snapshot/frame where this count was detected
        self.lane_number = lane_number
        self.skip_count = 0  # Track how many times this lane has been skipped

class Lanes:
    def __init__(self, lanes):
        self.lanes = lanes
    
    def getLanes(self):
        return self.lanes
    
    def lanesTurn(self):
        return self.lanes.pop(0)

    def enque(self, lane):
        self.lanes.append(lane)
    
    def lastLane(self):
        return self.lanes[len(self.lanes)-1]

def schedule(lanes):
    standard = 10  # Base time allocation
    min_threshold = 5  # Skip lanes with <= 5 vehicles
    max_skips = 2  # Allow a lane to be skipped at most 2 cycles
    
    turn = lanes.lanesTurn()

    if turn.count <= min_threshold:
        if turn.skip_count < max_skips:
            turn.skip_count += 1
            lanes.enque(turn)  
            return schedule(lanes)
        else:
            turn.skip_count = 0  

    total_cars = sum(lane.count for lane in lanes.getLanes() if lane.count > min_threshold) 
    reward = 0  

    for lane in lanes.getLanes():
        if total_cars > 0:
            lane_weight = lane.count / total_cars  
            reward += (turn.count - lane.count) * lane_weight  
    
    scheduled_time = max(5, round(standard + reward, 0))  
    lanes.enque(turn) 
    return scheduled_time

def display_result(wait_time, lanes):
    green = (0,255,0)
    red  = (0,0,255)
    yellow= (0,255,255)
     
    for i ,lane in enumerate(lanes.getLanes()):
        lane.frame = cv2.resize(lane.frame, (1280, 720))  # Resize to standard dimensions
        if wait_time <= 0 and (i == len(lanes.getLanes()) - 1 or i == 0):
            color = yellow
            text = "yellow:2 sec"
        elif wait_time >= 0 and i == len(lanes.getLanes()) - 1:
            color = green 
            text = "green:"+str(wait_time)+" sec"
        else:
            color = red
            text = "red:"+str(wait_time)+ " sec"
        
        # Adding text on frames
        lane.frame = cv2.putText(lane.frame, text, (60, 105), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 6)
        # lane.frame = cv2.putText(lane.frame, "vehicle count:"+str(lane.count), (60, 195), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)
        globals()['img%s' % lane.lane_number] = lane.frame

    # Concatenating images into a grid
    hori_image = np.concatenate((img1, img2), axis=1)
    hori2_image = np.concatenate((img3, img4), axis=1)
    all_lanes_image = np.concatenate((hori_image, hori2_image), axis=0)
    
    return all_lanes_image

def vehicle_count(boxes):
    vehicle = 0
    for box in boxes:
        if box.name == "car" or box.name == "truck" or box.name == "bus":
            vehicle += 1
    return vehicle

def postprocess(frame, outs, conf_threshold=0.5, nms_threshold=0.4):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Check if any indices are returned
    if len(indices) > 0:
        correct_boxes = []
        for i in indices.flatten():  # Flatten to avoid nested lists
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            box = BoundedBox(box[0], box[1], box[2], box[3], classIds[i], confidences[i])
            correct_boxes.append(box)
            frame = drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        return correct_boxes, frame
    else:
        print("No valid boxes detected")
        return [], frame

def drawPred(frame, classId, confidence, left, top, right, bottom):
    # Draw a bounding box on the frame
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), thickness=1)
    return frame

def final_output(net, output_layer, lanes):
    for lane in lanes.getLanes():
        lane.frame = cv2.resize(lane.frame, (416, 416))  # Resize to YOLOv3 input size
        blob = cv2.dnn.blobFromImage(lane.frame, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        
        # Forward pass through YOLOv3
        outs = net.forward(output_layer)
        
        # Postprocess the output
        boxes, frame = postprocess(lane.frame, outs)
        
        # Count vehicles
        count = vehicle_count(boxes)
        lane.count = count
        lane.frame = frame
        
    return lanes
