import numpy as np
import cv2 as cv
from control_detector_hough_lines import do_hough_image
from control_detector_common import DELIM_1, DELIM_4
from transform_color_mark import transform
from util import *


def map_angle_to_unit(angle):
    return np.interp(angle, [DELIM_1, DELIM_4], [0.0, 1.0])


def get_angles(frame):
    success, warped, _, _ = transform(frame)

    controls = get_control_positions("data/knobs_less.csv")

    for i, control in enumerate(controls):
        if i % 4 == 0:
            print()

        control_img = sub_image(frame, control)
        _, angle = do_hough_image(control_img, None, 164, 204, 14, 4)
        if angle is not None:
            position = map_angle_to_unit(angle)
            # print(angle*180/np.pi)
            print(angle)
        else:
            print("Unknown")


if __name__ == "__main__":
    cap = cv.VideoCapture("./data/camera_good.avi")
    _, frame = cap.read()
    get_angles(frame)
