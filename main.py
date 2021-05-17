import numpy as np
import cv2 as cv
from transform_color_mark import transform

# TODO calibration program?
MARKER_COLOR = np.array((177, 120, 194), dtype='uint8')

def main():
    cap = cv.VideoCapture(-1)  # Open the first camera connected to the computer.

    camera_matrix = np.loadtxt("camera_matrix.npy")
    distortion_coeff = np.loadtxt("distortion_coeff.npy")

    ret, frame = cap.read()
    h, w = frame.shape[:2]
    new_camera_matrix, camera_crop_region = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w,h), 1, (w,h))
    mapx, mapy = cv.initUndistortRectifyMap(camera_matrix, distortion_coeff, None, new_camera_matrix, (w,h), cv.CV_16SC2)
    
    while True:
        ret, frame = cap.read()
        frame = cv.undistort(frame, camera_matrix, distortion_coeff, None, new_camera_matrix)

        crop_x, crop_y, crop_w, crop_h = camera_crop_region
        frame = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

        success, transformed = transform(frame, draw_debug=True)
        if success:
            cv.imshow('transformed', transformed)

        if cv.waitKey(1) & 0xFF == ord('c'):
            cv.imwrite("transformed.png", transformed)
            print("wrote to file")

        cv.imshow('Camera frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

main()
