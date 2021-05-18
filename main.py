import numpy as np
import cv2 as cv
from transform_color_mark import transform

def main():
    cap = cv.VideoCapture(-1)  # Open the first camera connected to the computer.

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    #  camera_matrix = np.loadtxt("camera_matrix.npy")
    #  distortion_coeff = np.loadtxt("distortion_coeff.npy")
    camera_matrix = np.loadtxt("camera_matrix_720.npy")
    distortion_coeff = np.loadtxt("distortion_coeff_720.npy")

    ret, frame = cap.read()
    h, w = frame.shape[:2]
    new_camera_matrix, camera_crop_region = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w,h), 1, (w,h))
    mapx, mapy = cv.initUndistortRectifyMap(camera_matrix, distortion_coeff, None, new_camera_matrix, (w,h), cv.CV_16SC2)
    
    while True:
        ret, frame = cap.read()
        frame = cv.undistort(frame, camera_matrix, distortion_coeff, None, new_camera_matrix)

        crop_x, crop_y, crop_w, crop_h = camera_crop_region
        frame = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

        success, transformed = transform(frame, draw_debug=True, debug_scale=0.5)
        if success:
            cv.imshow('transformed', transformed)

        if cv.waitKey(1) & 0xFF == ord('c'):
            cv.imwrite("transformed.png", transformed)
            print("wrote to file")

        frame = cv.resize(frame, (int(frame.shape[1]*0.5), int(frame.shape[0]*0.5)));
        cv.imshow('Camera frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

main()
