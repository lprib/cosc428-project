import numpy as np
import cv2 as cv
from transform_color_mark import transform
from util import run_camera_loop, draw_resized

MOUSE_WINDOW_NAME = 'main'

def main_callback(frame, key, mouse_x, mouse_y):
    success, transformed, contour_img, thresh_img = transform(frame, draw_debug=True)
    if success:
        cv.imshow('transformed', transformed)
    draw_resized(contour_img, "contours", 0.5)

    if key == ord('c'):
        cv.imwrite("transformed.png", transformed)
        print("wrote to file")
    if key == ord('o'):
        cv.imwrite("camera.png", frame)
    if key == ord('m'):
        print(mouse_x, mouse_y)

    frame = cv.resize(frame, (int(frame.shape[1]*0.5), int(frame.shape[0]*0.5)));
    cv.imshow(MOUSE_WINDOW_NAME, frame)

run_camera_loop(1280, 720, "data/camera_matrix_720.npy", "data/distortion_coeff_720.npy", MOUSE_WINDOW_NAME, main_callback)
