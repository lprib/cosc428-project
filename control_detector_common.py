import cv2 as cv
import numpy as np


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


def draw_angle(drawing, angle, flipped):
    center = (int(drawing.shape[0] / 2), int(drawing.shape[1] / 2))
    cv.circle(drawing, center, 5, (255, 255, 255), 1)
    r = center[0]
    dims = (int(r*np.sin(angle)), int(r*np.cos(angle)))
    color = (0, 0, 255) if flipped else (0, 255, 0)
    cv.line(drawing, center, (center[0] + dims[0], center[1] + dims[1]), color, 2)
