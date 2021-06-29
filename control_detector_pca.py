import cv2 as cv
import numpy as np
from util import *
from control_detector_common import *

def do_pca_fit(img_orig, keys, cannyThreshold1, cannyThreshold2):
    img_gray = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(img_gray, cannyThreshold1, cannyThreshold2)
    mean, eigenvecs= cv.PCACompute(edges, mean=None)
    return img_gray, edges

def main_pca_fit():
    trackbar_info = [
        ("canny_thresh_1", 164, 255),
        ("canny_thresh_2", 204, 255)
    ]

    video = "./data/camera_moving.avi"
    control_detect_test_recorded_video(
        video,
        do_pca_fit,
        "PCA fit",
        trackbar_info,
        control_indices=None,
        draw_ref_img=True,
        write_to_video=False,
        draw_trackbars=True
    )

main_pca_fit()
