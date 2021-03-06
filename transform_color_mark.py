import numpy as np
import cv2 as cv

# offset to hue values to avoid wrap-around
HUE_ROTATION = 15

# hue thresholds (hue rotation is accounted for)
MARKER_THRESH_LOW = ((165 + HUE_ROTATION) % 180, 40, 50)
MARKER_THRESH_HIGH = ((11 + HUE_ROTATION) % 180, 255, 255)

# morph close kernel
MORPH_KERNEL = np.ones((5, 5), dtype=np.uint8)

# w, h to perspective warp into
PANEL_W = 1650
PANEL_H = 490

DP_EPSILON_COEFF = 0.01


def reshuffle_hull(hull_quad):
    """ Reshuffle such that the top left point is the 0th point in a hull, and resize to a (4, 2) array """
    hull_points = hull_quad.reshape((4, 2))

    # Find the point that is closest to the top left (0, 0)
    topleft_index = min(range(len(hull_points)),
                        key=lambda x: hull_points[x, 0]**2 + hull_points[x, 1]**2)
    # left shift the array such that the top-left point is at index 0
    hull_points = np.roll(hull_points, -topleft_index, axis=0)
    return hull_points


def get_quad_convex_hull(contours, contour_img):
    """ return a single bounding quad given a bunch of countours of color mark outlines"""

    # Create a single convex contour from concatenation of the contours
    contours_concat = np.vstack([contours[i] for i in range(len(contours))])
    contours_hull = cv.convexHull(contours_concat)

    # Simplify into (hopefully) a quadrilateral
    epsilon = DP_EPSILON_COEFF * cv.arcLength(contours_hull, True)
    hull_quad = cv.approxPolyDP(contours_hull, epsilon, True)

    cv.drawContours(contour_img, [contours_hull], -1, (0, 255, 0), 1)
    cv.drawContours(contour_img, [hull_quad], -1, (0, 0, 255), 1)

    if len(hull_quad) != 4:
        return None
    return reshuffle_hull(hull_quad)


def centroid(contour):
    """ get contour centroid from moments """
    moments = cv.moments(contour)
    # prevent div by zero
    if moments["m00"] == 0.0:
        return None
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return [cx, cy]


def get_quad_centroids(contours):
    """ get the centroids in correct order from contours, None if bad data """
    centroids = []
    for c in contours:
        cent = centroid(c)
        if cent is not None:
            centroids.append(cent)
    centroids = np.array(centroids)

    hull_quad = cv.convexHull(centroids)
    if len(hull_quad) != 4:
        return None
    else:
        return reshuffle_hull(hull_quad)


def transform(frame, draw_debug=False):
    """ perspective transform frame """
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Rotate hue values so it's possible to implement an inrange that wraps around to capture red
    frame_hsv[:, :, 0] += HUE_ROTATION
    frame_hsv[:, :, 0] %= 180
    # Mask any pixels that are within range of colored squares
    thresh_img_orig = cv.inRange(
        frame_hsv, MARKER_THRESH_LOW, MARKER_THRESH_HIGH)

    # Blur and morph open to remove noise
    thresh_img = cv.blur(thresh_img_orig, (5, 5))
    #  thresh_img = cv.morphologyEx(thresh_img, cv.MORPH_ERODE, MORPH_KERNEL, iterations=1)
    # thresh_img = cv.morphologyEx(
    # thresh_img, cv.MORPH_DILATE, MORPH_KERNEL, iterations=3)
    thresh_img = cv.morphologyEx(
        thresh_img, cv.MORPH_OPEN, MORPH_KERNEL, iterations=3)

    contours, hierarchy = cv.findContours(
        thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contour_img = frame.copy()
    if draw_debug:
        cv.drawContours(contour_img, contours, -1, (255, 0, 0), 2)

    # There should be four contours (ie. the outline of the four colored squares)
    if len(contours) != 4:
        return False, None, contour_img, thresh_img_orig

    #  quad_points = get_quad_convex_hull(contours, contour_img)
    quad_points = get_quad_centroids(contours)
    # Error if not a quad
    if quad_points is None:
        return False, None, contour_img, thresh_img_orig

    if draw_debug:
        for point in quad_points:
            cv.circle(contour_img, tuple(point), 5, (0, 255, 0), 2)

    quad_points = np.float32(quad_points)

    transform = cv.getPerspectiveTransform(quad_points, np.float32(
        [[0, 0], [PANEL_W, 0], [PANEL_W, PANEL_H], [0, PANEL_H]]))
    warped = cv.warpPerspective(frame, transform, (PANEL_W, PANEL_H))
    return True, warped, contour_img, thresh_img_orig
