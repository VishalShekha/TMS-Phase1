# AI-based-Traffic-Light-Control-System

Computer vision aided traffic light scheduling systems

AI-based-Traffic-Light-Control-System is an inteligent embedded system which applies computer vision to determine the density of cars at each lane on a traffic intersection so as to generate adaptive duration for green and red traffic light at each lane.

This repository represents an ongoing open source research into utilizing different object detection algorithims like YOLO to design an inteligent and adaptive traffic light control system. All the code and models are under research and development and subject to change.

Yolov5s is selected for this project due to its speed, lightness and accuracy. The yolov5s model can be found from https://github.com/ultralytics/yolov5

While the models speed is great, it is not efficent enough to be deployed on edge devices for inference. To take advantage of performance the model is exported into Onnx version and then exported to Tensorrt model which optimizes the model for inference. The performance of the model before and after optimization is shown below. Tutorials on how to export Yolov5s model into tensorrt model can be found on the tutorial section at https://github.com/ultralytics/yolov5

## Features

- Detect and counts vehicles from a camera feed on each lane
- Determine a green and red light duration based on comparison of each lanes vehicle density
- Displays a simulation

## Work flow

<p float="left">
  <img src="/screenshots/workflow.png" width="600" height=400 />

</p>

## Project directory

```
project
│   README.md
│   requirement.txt
│   main.py
|__ yolov3
│   │  coco.names
|   |  yolov3.cfg
|   |  yolov3.weigts
|__ data
    |  video.mp4
    │  video1.mp4
    |  video2.mp4
    |  video3.mp4
    |  video4.mp4
|__ algorithm
    |  utils.py
```

## Getting started

```sh
$ git clone https://github.com/Natnael-k/ubiquitous-octo-couscous.git
$ cd ubiquitous-octo-couscous
$ pip install requirement.txt
```

## How to run

For CPU and GPU environments...
The Onnx implementation can run both on CPU and GPU

```sh
$ cd implementation_with_yolov5s_onnx_model
$ python3 main.py  --sources video1.mp4,video2.mp4,video3.mp4,video5.mp4
```

Only for GPU environments...
The Tensorrt based implementation runs only on GPU

```sh
$ cd implementation_with_yolov5s_tensorrt_model
$ python3 main.py --sources video1.mp4,video2.mp4,video3.mp4,video5.mp4
```

## References

1. How to export yolov5s model to onnx:
   https://github.com/ultralytics/yolov5
2. How to export onnx model to tensorrt:
   https://github.com/SeanAvery/yolov5-tensorrt
