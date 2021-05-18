import numpy as np
import cv2 as cv


def run_camera_loop(cam_w, cam_h, camera_matrix_file, distortion_coeff_file, mouse_window_name, callback):
    cap = cv.VideoCapture(-1)  # Open the first camera connected to the computer.

    cap.set(cv.CAP_PROP_FRAME_WIDTH, cam_w)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cam_h)

    camera_matrix = np.loadtxt(camera_matrix_file)
    distortion_coeff = np.loadtxt(distortion_coeff_file)

    ret, frame = cap.read()
    h, w = frame.shape[:2]
    new_camera_matrix, camera_crop_region = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w,h), 1, (w,h))
    mapx, mapy = cv.initUndistortRectifyMap(camera_matrix, distortion_coeff, None, new_camera_matrix, (w,h), cv.CV_16SC2)


    mouse_x, mouse_y = 0, 0
    def mouse_callback(event, x, y, flags, param):
        # Closure capture:
        nonlocal mouse_x
        nonlocal mouse_y
        mouse_x, mouse_y = x, y

    cv.namedWindow(mouse_window_name)
    cv.setMouseCallback(mouse_window_name, mouse_callback)

    while True:
        ret, frame = cap.read()
        frame = cv.undistort(frame, camera_matrix, distortion_coeff, None, new_camera_matrix)

        crop_x, crop_y, crop_w, crop_h = camera_crop_region
        frame = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

        key = cv.waitKey(20) & 0xFF
        if key == ord('q'):
            break

        callback(frame, key, mouse_x, mouse_y)
