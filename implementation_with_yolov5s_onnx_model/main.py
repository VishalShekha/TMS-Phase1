import sys
import argparse
import pathlib
import cv2
import time
import os
import matplotlib.pyplot as plt

sys.path.insert(1, str(pathlib.Path.cwd().parents[0]) + "/common")
import utils as util
import numpy as np

# Define paths for YOLOv3 model
cfg_path = r"D:\\COding\\AI ka project\\ubiquitous-octo-couscous\\newer\\yolo\\yolov3.cfg"
weights_path = r"D:\\COding\\AI ka project\\ubiquitous-octo-couscous\\newer\\yolo\\yolov3.weights"
coco_names_path = r"D:\\COding\\AI ka project\\ubiquitous-octo-couscous\\newer\\coco.names"

# Ensure required files exist
if not all(os.path.exists(path) for path in [cfg_path, weights_path, coco_names_path]):
    raise FileNotFoundError("One or more required YOLOv3 files are missing!")

# Load YOLOv3 model
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
ln = net.getUnconnectedOutLayersNames()  # Get output layer names

def main(sources):
    # Read image from each lane's video source
    vs = cv2.VideoCapture(str(pathlib.Path.cwd().parents[0]) + "/datas/" + sources[0])
    vs2 = cv2.VideoCapture(str(pathlib.Path.cwd().parents[0]) + "/datas/" + sources[1])
    vs3 = cv2.VideoCapture(str(pathlib.Path.cwd().parents[0]) + "/datas/" + sources[2])
    vs4 = cv2.VideoCapture(str(pathlib.Path.cwd().parents[0]) + "/datas/" + sources[3])

    # Initial configuration of lanes
    lanes = util.Lanes([util.Lane("", "", 1), util.Lane("", "", 3), util.Lane("", "", 4), util.Lane("", "", 2)])
    wait_time = 0

    # Data storage for plotting (vehicle counts for each lane over time)
    lane_vehicle_counts = {1: [], 2: [], 3: [], 4: []}
    timestamps = []

    while True:
        # Read the next frame from the video sources
        (success, frame) = vs.read()
        (success, frame2) = vs2.read()
        (success, frame3) = vs3.read()
        (success, frame4) = vs4.read()

        # If the frame was not successfully captured, then we have reached the end
        if not success:
            break

        # Assign frames to corresponding lanes
        for i, lane in enumerate(lanes.getLanes()):
            if lane.lane_number == 1:
                lane.frame = frame
            elif lane.lane_number == 2:
                lane.frame = frame2
            elif lane.lane_number == 3:
                lane.frame = frame3
            elif lane.lane_number == 4:
                lane.frame = frame4

        # Process each frame through YOLOv3
        start = time.time()
        lanes = util.final_output(net, ln, lanes)  # Process frames through YOLOv3 model
        end = time.time()
        print("Total processing time:", str(end - start))

        # Track the vehicle count for each lane
        for lane in lanes.getLanes():
            lane_vehicle_counts[lane.lane_number].append(lane.count)

        timestamps.append(time.time())  # Record timestamp for this frame

        if wait_time <= 0:
            images_transition = util.display_result(wait_time, lanes)
            final_image = cv2.resize(images_transition, (1020, 720))
            cv2.waitKey(100)

            wait_time = util.schedule(lanes)  # Get waiting time for each lane

        # Display scheduled images
        images_scheduled = util.display_result(wait_time, lanes)
        final_image = cv2.resize(images_scheduled, (1020, 720))
        cv2.imshow("Traffic Management", final_image)
        cv2.waitKey(1)
        
        # After the loop ends, plot the graph
        # plot_vehicle_count_vs_time(lane_vehicle_counts, timestamps)
        wait_time -= 1  # Decrement wait time


def plot_vehicle_count_vs_time(lane_vehicle_counts, timestamps):
    # Convert timestamps to a more readable format
    times = [time - timestamps[0] for time in timestamps]  # Relative time from the first frame

    plt.figure(figsize=(10, 6))

    for lane_number, counts in lane_vehicle_counts.items():
        plt.plot(times, counts, label=f'Lane {lane_number}')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Vehicle Count')
    plt.title('Vehicle Count vs Time for Each Lane')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determines duration based on car count in images")
    parser.add_argument("--sources", help="Video feeds to be inferred on, the videos must reside in the datas folder", type=str, default="video1.mp4,video5.mp4,video2.mp4,video3.mp4")
    args = parser.parse_args()

    sources = args.sources.split(",")
    print(f"Using video sources: {sources}")
    main(sources)
