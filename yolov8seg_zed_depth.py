#!/usr/bin/env python3

# Import Python-native modules
import cv2
import numpy as np
import torch

# Import ZED SDK
import pyzed.sl as sl

# Import YOLOv8 related modules
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.augment import LetterBox


def print_available_objects(model_w_objects):
    print(f"------------------------------")
    print(f"Available objects:")
    print(f"------------------------------")
    for idx, name in enumerate(model_w_objects.names):
        print(f"{idx}:\t{model_w_objects.names[int(idx)]}")
    print(f"------------------------------")


def print_interested_objects(model_w_objects):
    print(f"\n------------------------------")
    print(f"Interested objects:")
    print(f"------------------------------")

    for idx, name in enumerate(model_w_objects.names):
        if model_w_objects.names[int(idx)] in interested_objects:
            print(f"{idx}:\t{model_w_objects.names[int(idx)]}")
    print(f"------------------------------")


if __name__ == "__main__":
    # Load a YOLOv8 model
    model = YOLO("yolov8s-seg.pt")  # Loading a pretrained segmentation model

    """
    When using the YOLO class, the model is automatically loaded onto the GPU
    If we were to print the model.device right here, it would say "cpu", but
    the model will automatically be transferred to the GPU when we perform a forward pass
    with the model; ex: results = model(image_cv2, verbose=False)
    If we were to print the model.device right after the forward pass, it would say "cuda:0"
    This behavior can be seen if we add the line: "print(f"Using {self.device.type} {self.device} for inference.")"
    to the following file on line 314: 
    <Installation path to anaconda3>\\anaconda3\envs\zed\Lib\site-packages\\ultralytics\engine\predictor.py
    """

    # Initialize ZED camera with particular parameters
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    # Open the camera (if it fails, try re-plugging in the ZED camera)
    # The ZED camera should show up under "Cameras" in Device Manager
    if not zed.open(init_params) == sl.ERROR_CODE.SUCCESS:
        print("***********************************")
        print("Error opening ZED camera (Possibly not detected). Exiting...")
        print("***********************************")
        exit(1)

    # Set runtime parameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()

    # Create a Mat to store depth data
    depth = sl.Mat()

    # Show object names that are available to detect (from model)
    print_available_objects(model)

    # Specify objects we're interested in (in list string-entry format)
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

            ####################################
            # Resize the image to the size similar to the masks
            # image_cv2 = cv2.resize(image_cv2, (640, 384))
            # ------------------------
            # Sizing breakdown
            # ------------------------
            # --> image_cv2.shape: (384, 640, 3)
            # --> image_cv2.shape: (720, 1280, 3) # If not resized
            ####################################

            # Perform YOLOv8 inference on the ZED image
            results = model(image_cv2, verbose=False)

            # Results plotting documentation
            # https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results.plot

            # Iterate through our results
            for r in results:
                # Create an annotator object (always us to more easily annotate our image prior to display)
                annotator = Annotator(image_cv2, line_width=2)

                ####################################
                # Get the bounding boxes and masks
                ####################################
                boxes = r.boxes
                masks = r.masks

                # Iterate through the bounding boxes, keeping track of the index
                # --> box_idx will be used to map the associated segmentation masks to each bounding box
                for box_idx, box in enumerate(boxes):
                    # Get the class label for the current bounding box
                    class_label = box.cls

                    # If the class label is not one of the interested objects, skip it
                    if model.names[int(class_label)] not in interested_objects:
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
                    # txt_color=(0,0,0) is black text
                    annotator.box_label(
                        box=b,
                        label=f"Class: {model.names[int(class_label)]} Depth: {depth_measurement_mm:.2f}m",
                        color=colors(class_label, True),
                        txt_color=(255, 255, 255),
                    )

                    # Get the segmentation mask for the current bounding box
                    mask = masks[box_idx]

                    ####################################
                    # In order to plot the segmentation mask, we follow a similar procedure to the procedure
                    # used in the YOLOv8 source code when using Results.plot()
                    # This methodology can be found in the section titled 'Plot SEgment results' in the following file:
                    # <Installation path to anaconda3>\\anaconda3\envs\zed\Lib\site-packages\\ultralytics\engine\results.py
                    # Specifically lines 243-250 from the above file...
                    ####################################
                    img = LetterBox(mask.shape[1:])(image=image_cv2)
                    im_gpu = (
                        torch.as_tensor(
                            img, dtype=torch.float16, device=mask.data.device
                        )
                        .permute(2, 0, 1)
                        .flip(0)
                        .contiguous()
                        / 255
                    )

                    # Plot the segmentation mask for the current bounding box
                    annotator.masks(
                        mask.data,
                        colors=[colors(x, True) for x in class_label],
                        im_gpu=im_gpu,
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
