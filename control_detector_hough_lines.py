import sys
import cv2 as cv
import numpy as np
from util import *
from control_detector_common import *


def get_start_end(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    return (x1, y1), (x2, y2)


def do_hough_image(img_orig, keys, cannyThreshold1, cannyThreshold2, houghThreshold, distance_power):
    img = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(img, cannyThreshold1, cannyThreshold2)
    lines = cv.HoughLines(edges, 1, np.pi/180, houghThreshold)
    lines_drawing = img_orig.copy()
    final_angle_drawing = img_orig.copy()

    angle = None
    if lines is not None:
        origin = (edges.shape[1] / 2, edges.shape[0] / 2)
        start_ends = [get_start_end(rho, theta) for ((rho, theta),) in lines]

        distances_to_origin = [distance_to_point(
            start, end, origin) for (start, end) in start_ends]
        max_dist = max(distances_to_origin)
        min_dist = min(distances_to_origin)

        # Use 0.01 as the lower bound here to avoid div-by-zero if there is only a single line
        # If there is only one line and it is assigned a weight of 0, the weighted average will not work
        distances_to_origin_norm = [
            np.interp(x, [min_dist, max_dist], [1.0, 0.01]) for x in distances_to_origin]
        distances_to_origin_norm = np.power(
            distances_to_origin_norm, distance_power)

        for (dist, (start, end)) in reversed(list(zip(distances_to_origin_norm, start_ends))):
            # Color lines based on distance from origin
            color = int(dist * 255)
            cv.line(lines_drawing, start, end, (255 - color, 0, color), 1)

        # Get list of just thetas
        thetas = lines[:, :, 1].reshape((-1))

        # Modular averaging (cite this!)
        # https://stackoverflow.com/questions/491738/how-do-you-calculate-the-average-of-a-set-of-circular-data
        avg_sin = np.dot(np.sin(thetas), distances_to_origin_norm)
        avg_cos = np.dot(np.cos(thetas), distances_to_origin_norm)
        avg_theta = np.arctan2(avg_sin, avg_cos)
        # Rotate such that theta is an angle around the origin of the line, not perpendicular line
        avg_theta = (np.pi - avg_theta) % np.pi

        left_handed = do_lr_detection(edges)
        if keys == ord("h"):
            print(left_handed)
        angle, flipped = do_angle_correction(avg_theta, left_handed)

        if keys == ord("a"):
            print(angle*180/np.pi)

        draw_angle(final_angle_drawing, angle, flipped)

    return (img_orig, gray(edges), lines_drawing, final_angle_drawing), angle


# same as above function, but discard the angle
def hough_shim(*args, **kwargs):
    return do_hough_image(*args, **kwargs)[0]


def main_hough():
    trackbar_info = [
        # ("canny_thresh_1", 59, 255),
        # ("canny_thresh_2", 144, 255),
        # ("hough_thresh", 10, 200),
        # ("distance_power", 3, 30)
        ("canny_thresh_1", 164, 255),
        ("canny_thresh_2", 204, 255),
        ("hough_thresh", 14, 200),
        ("distance_power", 4, 30)
    ]

    #  control_detect_test_static("data/transformed1.png", do_hough_image, "Hough lines", trackbar_info, control_indices=None)
    #  control_detect_test_video(do_hough_image, "Hough lines", trackbar_info, control_indices=range(10), draw_ref_img=True)
    # control_detect_test_video(
    # hough_shim, "Hough lines", trackbar_info, control_indices=None, draw_ref_img=True, write_to_video=True)
    if len(sys.argv) > 1 and sys.argv[1] == "moving":
        video = "./data/camera_moving.avi"
    else:
        video = "./data/camera_good.avi"

    control_detect_test_recorded_video(
        video,
        hough_shim,
        "Hough lines",
        trackbar_info,
        control_indices=None,
        draw_ref_img=True,
        write_to_video=False,
        draw_trackbars=True
    )


if __name__ == "__main__":
    main_hough()
