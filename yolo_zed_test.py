#!/usr/bin/env python3

# Import Python-native modules
import cv2
import numpy as np

# Import ZED SDK
import pyzed.sl as sl

# Import YOLOv8 related modules
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


def print_available_objects(model):
    print(f"------------------------------")
    print(f"Available objects:")
    print(f"------------------------------")
    for idx, name in enumerate(model.names):
        print(f"{idx}:\t{model.names[int(idx)]}")
    print(f"------------------------------")


def print_interested_objects(model):
    print(f"\n------------------------------")
    print(f"Interested objects:")
    print(f"------------------------------")
    for idx, name in enumerate(model.names):
        if model.names[int(idx)] in interested_objects:
            print(f"{idx}:\t{model.names[int(idx)]}")
    print(f"------------------------------")


if __name__ == "__main__":
    # Load a YOLOv8 model
    model = YOLO("yolov8s.pt")  # Loading a pretrained model

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

    # Show object names that are available to detect (from model)
    print_available_objects(model)

    # Specify objects we're interested in
    interested_objects = ["chair"]

    # Show interested object names and their indices in the model
    print_interested_objects(model)

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
            results = model(image_cv2, verbose=False)

            for r in results:
                # Create an annotator object
                annotator = Annotator(image_cv2, line_width=2)

                # Get the bounding boxes
                boxes = r.boxes
                for box in boxes:
                    # Get the class label
                    c = box.cls

                    # Filter out objects we're not interested in
                    if model.names[int(c)] not in interested_objects:
                        continue

                    # Get the bounding box coordinates
                    b = box.xyxy[0]

                    # Bounding box pixel locations (x1, y1, x2, y2)
                    bb_locations = box.xyxy[0].tolist()

                    # Convert bounding box pixel locations to integers
                    # (can't be used as pixel indices if they're floats)
                    for idx, b_val in enumerate(bb_locations):
                        bb_locations[idx] = int(b_val)

                    # Get the depth data within the bounding box
                    object_depth_data = depth_data[
                        bb_locations[1] : bb_locations[3],
                        bb_locations[0] : bb_locations[2],
                    ]

                    # Get the median depth value
                    median_depth = np.median(object_depth_data[object_depth_data > 0])

                    # Convert to meters (from mm)
                    depth_measurement_mm = median_depth / 1000.0

                    # Add a bounding box to the overall image surrounding the object
                    # Associated label depicts the class and depth measurement of the object
                    # color=(0,255,0) is green bounding box
                    # txt_color=(0,0,0) is black text
                    annotator.box_label(
                        box=b,
                        label=f"Class: {model.names[int(c)]} Depth: {depth_measurement_mm:.2f}m",
                        color=(0, 255, 0),
                        txt_color=(0, 0, 0),
                    )

            # Convert back to BGR for OpenCV Display
            image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

            # Display the image
            cv2.imshow("ZED with YOLOv8 and Depth", image_cv2)

            # Quit if q is pressed
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    zed.close()
    cv2.destroyAllWindows()
