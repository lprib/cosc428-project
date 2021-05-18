import cv2 as cv
import numpy as np
import sys
from util import get_control_positions, sub_image, control_detect_test_static, control_detect_test_video

def gray(img):
    return cv.cvtColor(img, cv.COLOR_GRAY2BGR)

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

def distance_to_point(start, end, point):
    return np.abs(
        (end[0] - start[0])*(start[1] - point[1]) - (start[0] - point[0])*(end[1] - start[1])
    ) / np.sqrt(
        (end[0] - start[0])*(end[0] - start[0]) + (end[1] - start[1])*(end[1] - start[1])
    )

def do_hough_image(img_orig, keys, cannyThreshold1, cannyThreshold2, houghThreshold):
    img = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(img, cannyThreshold1, cannyThreshold2)
    lines = cv.HoughLines(edges, 1, np.pi/180, houghThreshold)
    lines_drawing = img_orig.copy()
    final_angle_drawing = img_orig.copy()

    if lines is not None:
        origin = (edges.shape[1] / 2, edges.shape[0] / 2)
        start_ends = [get_start_end(rho, theta) for ((rho, theta),) in lines]

        distances_to_origin = [distance_to_point(start, end, origin) for (start, end) in start_ends]
        max_dist = max(distances_to_origin)
        min_dist = min(distances_to_origin)

        # Use 0.01 as the lower bound here to avoid div-by-zero if there is only a single line
        # If there is only one line and it is assigned a weight of 0, the weighted average will not work
        distances_to_origin_norm = [np.interp(x, [min_dist, max_dist], [1.0,0.01]) for x in distances_to_origin]

        for (dist, (start, end)) in zip(distances_to_origin_norm, start_ends):
            # Color lines based on distance from origin
            color = int(dist * 255)
            cv.line(lines_drawing, start, end, (255 - color, 0, color), 1)

        # Get list of just thetas
        thetas = lines[:,:,1].reshape((-1))

        # Modular averaging (cite this!)
        # https://stackoverflow.com/questions/491738/how-do-you-calculate-the-average-of-a-set-of-circular-data
        avg_sin = np.dot(np.sin(thetas), distances_to_origin_norm)
        avg_cos = np.dot(np.cos(thetas), distances_to_origin_norm)
        avg_theta = np.arctan(avg_sin / avg_cos)
        # Rotate such that theta is an angle around the origin of the line, not perpendicular line
        avg_theta = (np.pi - avg_theta) % np.pi

        left_handed = do_lr_detection(edges)
        if keys == ord("p"):
            print(left_handed)
        angle, flipped = do_angle_correction(avg_theta, left_handed)

        # draw line from center of image
        center = (int(lines_drawing.shape[0] / 2), int(lines_drawing.shape[1] / 2))
        cv.circle(lines_drawing, center, 5, (255, 255, 255), 1)
        r = center[0]
        dims = (int(r*np.sin(angle)), int(r*np.cos(angle)))
        color = (0, 0, 255) if flipped else (0, 255, 0)
        cv.line(final_angle_drawing, center, (center[0] + dims[0], center[1] + dims[1]), color, 2)

    return (img_orig, gray(edges), lines_drawing, final_angle_drawing)


def main_hough():
    trackbar_info = [
        ("canny_thresh_1", 59, 255),
        ("canny_thresh_2", 144, 255),
        ("hough_thresh", 10, 200)
    ]

    #  control_detect_test_static("data/transformed1.png", do_hough_image, "Hough lines", trackbar_info, control_indices=None)
    control_detect_test_video(do_hough_image, "Hough lines", trackbar_info, control_indices=range(10), draw_ref_img=True)

def get_sorted_contours(img, cannyThreshold1, cannyThreshold2):
    morph_kernel = np.ones((3, 3), np.uint8)

    edges = cv.Canny(img, cannyThreshold1, cannyThreshold2)
    morphed = cv.morphologyEx(edges, cv.MORPH_CLOSE, morph_kernel, iterations=2)

    contours, _hierarchy = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

    return edges, morphed, contours

def do_edge_morph(img_orig, keys, cannyThreshold1, cannyThreshold2):
    """ Cant get thresholds right """
    img = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)

    morph_kernel = np.ones((3, 3), np.uint8)

    contour_drawing = img_orig.copy()
    box = img_orig.copy()

    edges, morphed, contours = get_sorted_contours(img, cannyThreshold1, cannyThreshold2)

    if keys == ord("p"):
        print(do_lr_detection(morphed))

    if len(contours) > 0:
        cv.drawContours(contour_drawing, contours, 0, (255, 0, 0), 1)
        [vx, vy, x, y] = cv.fitLine(contours[0], cv.DIST_L2, 0, 0.1, 0.1)
        cols = img.shape[1]
        l = int((-x*vy/vx) + y)
        r = int(((cols-x)*vy/vx)+y)
        cv.line(box, (cols-1, r), (0, l), (0, 255, 0), 1)

    return (img_orig, gray(edges), gray(morphed), contour_drawing, box)

def main_edge_morph():
    trackbar_info = [
        ("canny_thresh_1", 59, 255),
        ("canny_thresh_2", 144, 255)
    ]
    control_detect_test_static("data/transformed2.png", do_edge_morph, "edge morph", trackbar_info)

def do_hough_p_image():
    pass

def main_hough_p():
    trackbar_info = (
        ('CannyThreshold1', 59, 255),
        ('CannyThreshold2', 144, 255),
        ("HoughThreshold", 10, 200)
    )

    control_detect_test_static("data/transformed2.png", do_hough_p_images, "probabilistic hough lines", trackbar_info)

# Returns True if right handed, False if left handed
def do_lr_detection(edges):
    center_x = edges.shape[1] // 2
    left = np.sum(edges[:, :center_x])
    right = np.sum(edges[:, center_x:])

    return left > right

# assumes angle between 0 and pi
def do_angle_correction(angle, left_handed):
    delim_1 = 1 * np.pi / 4 - 0.4
    delim_4 = 7 * np.pi / 4 + 0.4
    delim_2 = delim_4 - np.pi
    delim_3 = delim_1 + np.pi
    flipped = False

    if angle <= delim_1 or angle >= delim_4:
        # always flip if in bottom quadrant
        angle += np.pi
        flipped = True
    elif angle >= delim_1 and angle <= delim_2 and left_handed:
        # if right handed but angle is in left quadrant
        angle += np.pi
        flipped = True
    elif angle >= delim_3 and angle <= delim_4 and not left_handed:
        # if left handed but angle is in right quadrant
        angle += np.pi
        flipped = True

    return angle % (2 * np.pi), flipped


main_hough()
#  main_erosion()
#  main_edge_morph()
