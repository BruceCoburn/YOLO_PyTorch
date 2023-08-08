import cv2
import pyzed.sl as sl
import torch
import numpy as np

import sys
# sys.path.append('C:\\yolov5\\')

# from models.experimental import attempt_load

from ultralytics import YOLO

from ultralytics.utils.plotting import Annotator

# Load a YOLOv8 model
# model = YOLO('yolov8n.pt')  # Loading a pretrained model
model = YOLO('yolov8s.pt')  # Loading a pretrained model

# Initialize ZED camera with particular parameters
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30

# Open the camera
if not zed.open(init_params) == sl.ERROR_CODE.SUCCESS:
    exit(1)

runtime_parameters = sl.RuntimeParameters()

# Create a Mat to store depth data
depth = sl.Mat()

# Specify objects we're interested in
interested_objects = ['chair']
print(f'Interested objects: {interested_objects}')
"""
for interested_object in interested_objects:
    print(f'reg str: {interested_object}')
    print(f'interested_object: {model.names[interested_object]}')
"""

for name in model.names:
    print(f'name: {name}')

# Main loop
while True:
    # Grab a new frame
    err = zed.grab(runtime_parameters)
    if err == sl.ERROR_CODE.SUCCESS:
        # Retrieve the left image
        image_zed = sl.Mat()
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        image_cv2 = image_zed.get_data()

        #################################################################
        # NECESSARY FOR YOLOv8: Convert to RGB (when using np.ndarray)
        #################################################################
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

        # Retrieve depth measurements, store pixel depth data in Mat
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        depth_data = depth.get_data()

        # Perform YOLOv8 inference on the ZED image
        results = model(image_cv2)

        # print(f'results: {results}')
        #for name in results.names:
        #    print(f'result name: {name}')

        for r in results:

            # Create an annotator object
            annotator = Annotator(image_cv2, line_width=2)

            # Get the bounding boxes
            boxes = r.boxes
            for box in boxes:

                # Get the bounding box coordinates
                b = box.xyxy[0]

                # Get the class label
                c = box.cls

                # print(f'c: {c}')

                # Bounding box pixel locations
                bb_locations = box.xyxy[0].tolist()

                # Convert bounding box pixel locations to integers (can't be floats)
                for idx, b_val in enumerate(bb_locations):
                    bb_locations[idx] = int(b_val)

                # Get the depth data within the bounding box
                object_depth_data = depth_data[bb_locations[1]:bb_locations[3], bb_locations[0]:bb_locations[2]]

                # Get the median depth value
                median_depth = np.median(
                    object_depth_data[object_depth_data > 0])

                # Convert to meters (from mm)
                median_depth_meters = median_depth / 1000.0

                # Add a bounding box to the overall image surrounding the object
                # Associated label depicts the class and depth measurement of the object
                # color=(0,255,0) is green bounding box
                # txt_color=(0,0,0) is black text
                annotator.box_label(box=b, label=f"Class: {model.names[int(c)]} Depth: {median_depth_meters:.2f}m",
                                    color=(0, 255, 0), txt_color=(0, 0, 0))

        # Convert back to BGR for OpenCV Display
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

        # Display the image
        cv2.imshow('ZED with YOLOv8 and Depth', image_cv2)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

zed.close()
cv2.destroyAllWindows()
