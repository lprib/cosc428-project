import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from control_detector_hough_lines import do_hough_image
from control_detector_contour_fit import do_contour_fit
from util import *

HOUGH_THRESH = 14
CANNY1 = 49
CANNY2 = 102

def hough_stat_func(crop):
    return do_hough_image(crop, None, CANNY1, CANNY2, HOUGH_THRESH, 4)[1]

def contour_stat_func(crop):
    return do_contour_fit(crop, None, CANNY1, CANNY2)[1]

def compute_sample_matrix(test_function, vidfile):
    cap = cv.VideoCapture(vidfile)
    controls = get_control_positions("./data/knobs_more.csv")

    samples = [[] for _ in range(len(controls))]

    frame_num = 0
    while True:
        print(f"calculating frame {frame_num}")
        frame_num += 1

        ret, frame = cap.read()
        if not ret:
            break

        success, transformed, _, _ = transform(frame, draw_debug=False)
        if success:
            for control_idx, control in enumerate(controls):
                crop = sub_image(transformed, control)
                angle = test_function(crop)
                if angle is not None:
                    samples[control_idx].append(angle)
    return samples

def write_to_file(samples, filename):
    with open(filename, "w") as file:
        for control_list in samples:
            file.write(" ".join(map(str, control_list)))
            file.write("\n")

def compute_hough_sample(power, vidfile, vidname):
    print(f"Computing hough samples power {power}, vid {vidname}")
    filename = f"./outputs/samples_hough_power{power}_{vidname}.dat"
    def func(crop):
        return do_hough_image(crop, None, CANNY1, CANNY2, HOUGH_THRESH, power)[1]
    samples = compute_sample_matrix(func, vidfile)
    write_to_file(samples, filename)



def compute_all_samples():
    compute_hough_sample(0, "./data/camera_good.avi", "stillcam")
    compute_hough_sample(3, "./data/camera_good.avi", "stillcam")
    compute_hough_sample(10, "./data/camera_good.avi", "stillcam")

    compute_hough_sample(0, "./data/camera_moving.avi", "movecam")
    compute_hough_sample(3, "./data/camera_moving.avi", "movecam")
    compute_hough_sample(10, "./data/camera_moving.avi", "movecam")

    print("Computing samples for contour fit")
    samples = compute_sample_matrix(contour_stat_func, "./data/camera_good.avi")
    print("Writing contour to file")
    write_to_file(samples, "./outputs/samples_contour_0_stillcam.dat")

    print("Computing samples for contour fit")
    samples = compute_sample_matrix(contour_stat_func, "./data/camera_moving.avi")
    print("Writing contour to file")
    write_to_file(samples, "./outputs/samples_contour_0_movecam.dat")

def read_sample_file(filename):
    with open(filename, "r") as file:
        samples = [list(map(float, line.split())) for line in file.readlines()]
    return samples

def get_truth_values():
    return list(map(float, open("./data/truth_angles.dat", "r").readlines()))

def error_histogram(samples, title):
    errors = []
    for truth, control_list in zip(get_truth_values(), samples):
        errors.extend(samp - truth for samp in control_list)
    errors = np.array(errors)
    stddev = np.std(errors)
    mean = np.mean(errors)
    print(f"stddev {stddev} mean {mean}")
    plt.figure()
    plt.hist(errors, bins=300)
    plt.title(title)
    plt.grid()

#  compute_all_samples()

samples = read_sample_file("./outputs/samples_hough_power0_movecam.dat")
error_histogram(samples, "power 0")
samples = read_sample_file("./outputs/samples_hough_power3_movecam.dat")
error_histogram(samples, "power 3")
samples = read_sample_file("./outputs/samples_hough_power10_movecam.dat")
error_histogram(samples, "power 10")
plt.show()

