import numpy as np
import cv2

import os

class BoundedBox:
    def __init__(self, xmin, ymin, xmax, ymax, ids, confidence):
        coco_path = os.path.join(os.getcwd(), 'yolo', 'coco.names')

        if not os.path.exists(coco_path):
            raise FileNotFoundError(f"coco.names file not found at {coco_path}")

        with open(coco_path, 'r') as f:
            self.classes = f.read().strip().splitlines()

        if not (0 <= ids < len(self.classes)):
            raise ValueError("Invalid class ID")

        self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
        self.name = self.classes[ids]
        self.confidence = confidence

class Lane:
    def __init__(self, count, frame, lane_number):
        self.count = count
        self.frame = frame
        self.lane_number = lane_number
        self.skip_count = 0

class Lanes:
    def __init__(self, lanes):
        self.lanes = lanes

    def get_lanes(self):
        return self.lanes

    def rotate_lane(self):
        return self.lanes.pop(0)

    def enqueue(self, lane):
        self.lanes.append(lane)

    def last_lane(self):
        return self.lanes[-1]

def schedule(lanes, base_time=10, min_threshold=5, max_skips=2):
    turn = lanes.rotate_lane()

    if turn.count <= min_threshold and turn.skip_count < max_skips:
        turn.skip_count += 1
        lanes.enqueue(turn)
        return schedule(lanes)

    total_cars = sum(lane.count for lane in lanes.get_lanes() if lane.count > min_threshold)

    reward = sum((turn.count - lane.count) * (lane.count / total_cars) for lane in lanes.get_lanes() if total_cars > 0)

    turn.skip_count = 0
    lanes.enqueue(turn)

    return max(5, round(base_time + reward))

def display_result(wait_time, lanes):
    colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255)]

    for i, lane in enumerate(lanes.get_lanes()):
        lane.frame = cv2.resize(lane.frame, (1280, 720))

        if wait_time <= 0 and i == 0:
            color, text = colors[2], "yellow:2 sec"
        elif wait_time > 0 and i == len(lanes.get_lanes()) - 1:
            color, text = colors[0], f"green:{wait_time} sec"
        else:
            color, text = colors[1], f"red:{wait_time} sec"

        cv2.putText(lane.frame, text, (60, 105), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 6)

    return np.concatenate([
        np.concatenate((lanes.get_lanes()[0].frame, lanes.get_lanes()[1].frame), axis=1),
        np.concatenate((lanes.get_lanes()[2].frame, lanes.get_lanes()[3].frame), axis=1)
    ], axis=0)

def vehicle_count(boxes):
    return sum(1 for box in boxes if box.name in {"car", "truck", "bus"})

def postprocess(frame, outs, conf_threshold=0.5, nms_threshold=0.4):
    frame_height, frame_width = frame.shape[:2]
    boxes, class_ids, confidences = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                center_x, center_y, width, height = (detection[0:4] * [frame_width, frame_height, frame_width, frame_height]).astype(int)
                left, top = center_x - width // 2, center_y - height // 2

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, left + width, top + height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if len(indices) == 0:
        return [], frame

    correct_boxes = [BoundedBox(*boxes[i], class_ids[i], confidences[i]) for i in indices.flatten()]

    for box in correct_boxes:
        cv2.rectangle(frame, (box.xmin, box.ymin), (box.xmax, box.ymax), (255, 0, 0), 1)

    return correct_boxes, frame

def final_output(net, output_layer, lanes):
    for lane in lanes.get_lanes():
        resized_frame = cv2.resize(lane.frame, (416, 416))

        blob = cv2.dnn.blobFromImage(resized_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        outs = net.forward(output_layer)
        boxes, lane.frame = postprocess(lane.frame, outs)

        lane.count = vehicle_count(boxes)

    return lanes
