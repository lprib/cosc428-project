import cv2 as cv
import numpy as np
from util import *
from control_detector_common import *


def get_sorted_contours(img, cannyThreshold1, cannyThreshold2):
    morph_kernel = np.ones((3, 3), np.uint8)

    edges = cv.Canny(img, cannyThreshold1, cannyThreshold2)
    morphed = cv.morphologyEx(edges, cv.MORPH_CLOSE,
                              morph_kernel, iterations=2)

    contours, _hierarchy = cv.findContours(
        morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

    return edges, morphed, contours


def do_contour_fit(img_orig, keys, cannyThreshold1, cannyThreshold2):
    """ Cant get thresholds right """
    img = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)

    morph_kernel = np.ones((3, 3), np.uint8)

    contour_drawing = img_orig.copy()
    line_drawing = img_orig.copy()

    edges, morphed, contours = get_sorted_contours(
        img, cannyThreshold1, cannyThreshold2)

    angle = None
    if len(contours) > 0:
        cv.drawContours(contour_drawing, contours, 0, (255, 0, 0), 1)
        [vx, vy, x, y] = cv.fitLine(contours[0], cv.DIST_L2, 0, 0.1, 0.1)
        cols = img.shape[1]

        angle = np.arctan2(vx, vy)[0]
        left_handed = do_lr_detection(edges)
        angle, flipped = do_angle_correction(angle, left_handed)
        draw_angle(line_drawing, angle, flipped)

    return (img_orig, gray(edges), gray(morphed), contour_drawing, line_drawing), angle


def contour_fit_shim(*args, **kwargs):
    return do_contour_fit(*args, **kwargs)[0]



def main_contour_fit():
    trackbar_info = [
        ("canny_thresh_1", 164, 255),
        ("canny_thresh_2", 204, 255)
    ]
    #  control_detect_test_static("data/transformed2.png", do_contour_fit, "edge morph", trackbar_info)
    
    #  control_detect_test_video(do_contour_fit, "edge morph",
                              #  trackbar_info, control_indices=range(10), draw_ref_img=True)
    video = "./data/camera_moving.avi"
    control_detect_test_recorded_video(
        video,
        contour_fit_shim,
        "Coutour fit",
        trackbar_info,
        control_indices=None,
        draw_ref_img=True,
        write_to_video=False,
        draw_trackbars=True
    )

if __name__ == "__main__":
    main_contour_fit()
