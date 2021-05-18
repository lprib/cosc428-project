import cv2 as cv
import numpy as np
import sys
from util import get_control_positions, sub_image, control_detect_test

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

def do_hough_image(img_orig, cannyThreshold1, cannyThreshold2, houghThreshold):
    drawing = img_orig.copy()
    img = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(img, cannyThreshold1, cannyThreshold2)
    lines = cv.HoughLines(edges, 1, np.pi/180, houghThreshold)

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
            color = int(dist * 255)
            cv.line(drawing, start, end, (255 - color, 0, color), 1)

        # Get list of just thetas
        thetas = lines[:,:,1].reshape((-1)) + np.pi * 2
        avg_theta = np.average(thetas, weights=distances_to_origin_norm)
        # Rotate such that theta is an angle around the origin of the line, not perpendicular line
        avg_theta = (np.pi - avg_theta) % np.pi

        # draw line from center of image
        center = (int(drawing.shape[0] / 2), int(drawing.shape[1] / 2))
        cv.circle(drawing, center, 5, (255, 255, 255), 1)
        r = center[0]
        dims = (int(r*np.sin(avg_theta)), int(r*np.cos(avg_theta)))
        cv.line(drawing, center, (center[0] + dims[0], center[1] + dims[1]), (0, 255, 0), 1)

    return (img_orig, cv.cvtColor(edges, cv.COLOR_GRAY2BGR), drawing)


def main_hough():
    trackbar_info = [
        ("canny_thresh_1", 59, 255),
        ("canny_thresh_2", 144, 255),
        ("hough_thresh", 10, 200)
    ]

    control_detect_test("Hough lines", trackbar_info, "data/transformed1.png", do_hough_image, control_indices=[3])

def main_erosion():
    img = cv.imread("single_knob.png")
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.namedWindow('erosion angle')
    cv.createTrackbar('adaptive C', 'erosion angle', 0, 255, lambda x: None)

    while True:
        thresh_c = cv.getTrackbarPos('adaptive C', 'erosion angle')
        blurred = cv.GaussianBlur(img_gray, (7, 7), 0)
        ret, thresh = cv.threshold(blurred, thresh_c, 255, cv.THRESH_BINARY)
        #  thresh = cv.adaptiveThreshold(blurred
        combined = np.concatenate((img_gray, blurred, thresh), axis=1)
        cv.imshow('erosion angle', combined)
        if cv.waitKey(1000) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break

def edge_morph(img_orig, cannyThreshold1, cannyThreshold2):
    """ Cant get thresholds right """
    img = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)

    morph_kernel = np.ones((3, 3), np.uint8)

    edges = cv.Canny(img, cannyThreshold1, cannyThreshold2)
    morphed = cv.morphologyEx(edges, cv.MORPH_CLOSE, morph_kernel, iterations=2)

    contours, _hierarchy = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

    contour_drawing = img_orig.copy()
    box = img_orig.copy()
    if len(contours) > 0:
        largest = contours[0]
        cv.drawContours(contour_drawing, contours, 0, (0, 255, 0), 1)
        [vx, vy, x, y] = cv.fitLine(largest, cv.DIST_L2, 0, 0.1, 0.1)
        cols = img.shape[1]
        l = int((-x*vy/vx) + y)
        r = int(((cols-x)*vy/vx)+y)
        cv.line(box, (cols-1, r), (0, l), (0, 255, 0), 1)

    return (img_orig, cv.cvtColor(edges, cv.COLOR_GRAY2BGR), cv.cvtColor(morphed, cv.COLOR_GRAY2BGR), contour_drawing, box)

def main_edge_morph():
    trackbar_info = [
        ("canny_thresh_1", 59, 255),
        ("canny_thresh_2", 144, 255)
    ]
    control_detect_test("edge morph", trackbar_info, "data/transformed2.png", edge_morph)

def do_hough_p_image():
    pass

def main_hough_p():
    trackbar_info = (
        ('CannyThreshold1', 59, 255),
        ('CannyThreshold2', 144, 255),
        ("HoughThreshold", 10, 200)
    )

    control_detect_test("probabilistic hough lines", trackbar_info, "data/transformed2.png", do_hough_p_image)

main_hough()
#  main_erosion()
#  main_edge_morph()
