
import numpy as np
import cv2
import time
import pathlib


"""
a blueprint for a bounded box with its corresponding name,confidence score and 
"""
print(pathlib.Path.cwd())

class BoundedBox:
    
    def __init__(self, xmin, ymin, xmax, ymax, ids, confidence):
        with open(str(pathlib.Path.cwd().parents[0])+"/datas/coco.names", 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')  # stores a list of classes
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax       
        self.ymax = ymax
        self.name = self.classes[ids]
        self.confidence = confidence   


"""
a blueprint that has lanes as lists and give queue like functionality 
to reorder lanes based on their turn for green and red light state
"""

class Lanes:
    def __init__(self,lanes):
        self.lanes=lanes
    
    def getLanes(self):
        
        return self.lanes
    
    def lanesTurn(self):
        
       return self.lanes.pop(0)

    def enque(self,lane):
 
       return self.lanes.append(lane)
    def lastLane(self):
       return self.lanes[len(self.lanes)-1]
"""
a blueprint that has lanes as lists and give queue like functionality 
to reorder lanes based on their turn for green and red light state
"""
# class Lane:
#     def __init__(self,count,frame,lane_number):
#         self.count = count
#         self.frame = frame
#         self.lane_number = lane_number

class Lane:
    def __init__(self, count, frame, lane_number):
        self.count = count  # Number of vehicles in the lane
        self.frame = frame  # Snapshot/frame where this count was detected
        self.lane_number = lane_number
        self.skip_count = 0  # Track how many times this lane has been skipped


    
"""
given lanes object return a duration based on comparison of each lane vehicle count
"""
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

       
"""
given duration and lanes, returns a grid image containing frames of each lane with
their corresponding waiting duration
"""   

def display_result(wait_time,lanes):
    green = (0,255,0)
    red  = (0,0,255)
    yellow= (0,255,255)
     
    for i ,lane in enumerate(lanes.getLanes()):
        #resized so that all images have the same dimension inorder to be concatenable
        lane.frame = cv2.resize(lane.frame,(1280, 720)) 
        
        if(wait_time<=0 and (i==(len(lanes.getLanes())-1) or i==0)):
           color=yellow
           text="yellow:2 sec"
           
        elif(wait_time>=0 and i==(len(lanes.getLanes())-1)):
            color = green 
            text="green:"+str(wait_time)+" sec"
        
        else:
            color=red
            text="red:"+str(wait_time)+ " sec"
        lane.frame = cv2.putText(lane.frame,text,(60,105),cv2.FONT_HERSHEY_SIMPLEX,4,color,6)
        lane.frame = cv2.putText(lane.frame,"vehicle count:"+str(lane.count),(60,195),cv2.FONT_HERSHEY_SIMPLEX,3,color,5)
        globals()['img%s' % lane.lane_number]=lane.frame
        


    hori_image = np.concatenate((img1, img2), axis=1)
    hori2_image = np.concatenate((img3, img4), axis=1)
    all_lanes_image = np.concatenate((hori_image, hori2_image), axis=0)
    
    return all_lanes_image

# import cv2
# import numpy as np

# def display_result(wait_time, lanes):
#     # Define signal colors
#     COLORS = {
#         "green": (0, 255, 0),
#         "red": (0, 0, 255),
#         "yellow": (0, 255, 255)
#     }

#     lane_images = []  # List to store lane images

#     for i, lane in enumerate(lanes.getLanes()):
#         lane.frame = cv2.resize(lane.frame, (1280, 720))  # Standardized frame size

#         # Determine signal color and text
#         if wait_time <= 0 and (i == 0 or i == len(lanes.getLanes()) - 1):
#             color, text = COLORS["yellow"], "Yellow: 2 sec"
#         elif wait_time > 0 and i == len(lanes.getLanes()) - 1:
#             color, text = COLORS["green"], f"Green: {wait_time} sec"
#         else:
#             color, text = COLORS["red"], f"Red: {wait_time} sec"

#         # Overlay lane number at the top
#         cv2.putText(lane.frame, f"Lane {lane.lane_number}", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)

#         # Overlay traffic light signal and vehicle count
#         cv2.putText(lane.frame, text, (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 6)
#         cv2.putText(lane.frame, f"Vehicle Count: {lane.count}", (60, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)

#         lane_images.append(lane.frame)  # Store the modified lane frame

#     # Arrange images dynamically into rows
#     num_cols = 2  # Define number of columns per row
#     row_images = [np.concatenate(lane_images[i:i + num_cols], axis=1) for i in range(0, len(lane_images), num_cols)]
#     final_image = np.concatenate(row_images, axis=0) if len(row_images) > 1 else row_images[0]

#     return final_image





# given detecteed boxes, return number of vehicles on each box
def vehicle_count(Boxes):
        vehicle=0
        for box in Boxes:
            if box.name == "car" or box.name == "truck" or box.name == "bus":
                vehicle=vehicle+1  

        return vehicle

# given the grid dimension, returns a 2d grid
def _make_grid(nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

def drawPred( frame, classId, conf, left, top, right, bottom):
        
       
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), thickness=6)

        return frame
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    ratioh, ratiow = frameHeight / 320, frameWidth / 320
    classIds = []
    confidences = []
    boxes = []

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > 0.5 and detection[4] > 0.5:
                center_x = int(detection[0] * ratiow)
                center_y = int(detection[1] * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    
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

"""
interpret the ouptut boxes into the appropriate bounding boxes based on the yolo paper 
logspace transform
"""
def modify(outs,confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):
        print(str(pathlib.Path.cwd().parents[0])+"/datas")
        with open(str(pathlib.Path.cwd().parents[0])+'/datas/coco.names', 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')   
        print("dir:"+str(pathlib.Path.cwd()))
        colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(classes))]
        num_classes = len(classes)
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        nl = len(anchors)
        na = len(anchors[0]) // 2
        no = num_classes + 5
        grid = [np.zeros(1)] * nl
        stride = np.array([8., 16., 32.])
        anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, 1, -1, 1, 1, 2)

        
        z = []  # inference output
        for i in range(nl):
            bs, _, ny, nx,c = outs[i].shape  
            if grid[i].shape[2:4] != outs[i].shape[2:4]:
                grid[i] = _make_grid(nx, ny)
                

            y = 1 / (1 + np.exp(-outs[i])) 
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * int(stride[i])
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1,no))
        z = np.concatenate(z, axis=1)
        return z


"""
given each lanes image, it inferences using trt engine on the image, return lanes object
containg processed image and waiting duration for each image

"""
# def final_output_tensorrt(processor,lanes):
     
#     for lane in lanes.getLanes():
#             lane.frame=cv2.resize(lane.frame,(1280,720))      #resize into a standard image  dimension
#             start = time.time()
#             output = processor.detect(lane.frame)
#             end = time.time() 
#             print("fps:"+str(end-start))   
#             dets = modify(output)
#             boxes,frame = postprocess(lane.frame,dets)
#             count = vehicle_count(boxes)
#             lane.count= count
#             lane.frame=frame
            
        
        
#     return lanes

"""
given each lanes image, it inferences onnx model on the image, return lanes object
containg processed image and waiting duration for each image

"""

def final_output(net,output_layer,lanes):
        
        for lane in lanes.getLanes():
            lane.frame=cv2.resize(lane.frame,(1280,720))      #resize into a standard image  dimension
            blob = cv2.dnn.blobFromImage(lane.frame, 1 / 255.0, (320, 320),
                swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(output_layer)
            end = time.time() 
            print("fps:"+str(end-start))   
            dets = modify(layerOutputs)
            boxes,frame = postprocess(lane.frame,dets)
            start = time.time()
            count = vehicle_count(boxes)
            lane.count= count
            lane.frame=frame
            
            
        
        
        return lanes
