import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from control_detector_hough_lines import do_hough_image
from util import *


def do_stat_test(distance_power):
    angles = []

    control = get_control_positions("data/knobs_more.csv")[1]

    def main_callback(frame, key, mouse_x, mouse_y):
        nonlocal angles
        succ, warped, contour, thresh = transform(frame, draw_debug=True)
        draw_resized(contour, "contour", 0.3)
        draw_resized(thresh, "thresh", 0.3)
        if succ:
            control_img = sub_image(warped, control)
            images, angle = do_hough_image(
                control_img, -1, 49, 144, 10, distance_power)
            cv.imshow("main", np.concatenate(images, axis=1))
            if key == ord(' '):
                angles.append(angle)
                print(len(angles))

    run_camera_loop(1280, 720, "data/camera_matrix_720.npy",
                    "data/distortion_coeff_720.npy", "main", main_callback)

    print(f"writing file for {distance_power}")
    with open(f"outputs/angles_power_{distance_power}.dat", "w") as f:
        for n in angles:
            f.write(f"{n}\n")


def analyse(distance_power, subplot):
    TRUTH = 3 * np.pi / 4 - (5/180*np.pi)
    filename = f"outputs/angles_power_{distance_power}.dat"
    angles = np.array([float(x.strip())
                      for x in open(filename).readlines() if x != "None\n"])
    errors_deg = (angles - TRUTH) / np.pi * 180
    avg_error_deg = np.average(np.abs(errors_deg))
    print(f"stats for distance power {distance_power}")
    print(f"avg error {avg_error_deg}")
    subplot.hist(errors_deg, bins=10)
    subplot.set_title(f"Error for power={distance_power}")
    subplot.set_ylabel("count")
    subplot.set_xlabel("angle (deg)")
    #  hist =

#  do_stat_test(0)
#  do_stat_test(3)
#  do_stat_test(10)


fig, (a1, a2, a3) = plt.subplots(3, 1, sharex=True)
analyse(0, a1)
analyse(3, a2)
analyse(10, a3)
plt.tight_layout()
plt.show()
