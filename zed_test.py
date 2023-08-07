import pyzed.sl as sl
import cv2

def main():
    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters for the ZED
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720   # Use HD720 video mode
    init_params.camera_fps = 60  # Set fps at 60

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    try:
        while True:
            # Grab an image from the ZED camera
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Retrieve the left image
                zed.retrieve_image(image, sl.VIEW.LEFT)
                # Convert the image to a format that can be displayed with OpenCV
                image_ocv = image.get_data()
                cv2.imshow("ZED", image_ocv)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        zed.close()

if __name__ == "__main__":
    main()
