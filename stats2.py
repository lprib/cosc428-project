#  import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from control_detector_hough_lines import do_hough_image
from control_detector_contour_fit import do_contour_fit
from util import *

if 1:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

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
    mean = np.mean(np.abs(errors))
    print(f"{title} stddev {stddev} mean {mean}")
    #  plt.figure()
    #  plt.hist(errors, bins=300)
    #  plt.title(title)
    #  plt.grid()

def error_histogram_ax(ax, samples, title):
    errors = []
    for truth, control_list in zip(get_truth_values(), samples):
        errors.extend(samp - truth for samp in control_list)
    errors = np.array(errors)
    ax.hist(errors, bins=500)
    ax.set_title(title)
    #  ax.set_xlabel("Error in detected angle (radians)")
    #  ax.set_ylabel("Count")
    ax.set_xlim(-1, 1)
    ax.xaxis.set_ticks(np.linspace(-4, 4, 11))
    ax.set_xlabel("Error in angle (radians)")
    ax.set_ylabel("Count")
    ax.grid()


def samples_within_1pc(samples):
    count = 0
    total = 0
    for truth, control_list in zip(get_truth_values(), samples):
        for x in control_list:
            total += 1
            if abs(x-truth) < 5*truth/100:
                count += 1
    print(f"{count} / {total}")



def power_comparision_plot():
    fig, ((a1, a2, a3, a4)) = plt.subplots(4, 1, sharex=True, sharey=True)

    samples = read_sample_file("./outputs/samples_hough_power0_movecam.dat")
    error_histogram_ax(a1, samples, "Hough lines, no power weighting")

    samples = read_sample_file("./outputs/samples_hough_power3_movecam.dat")
    error_histogram_ax(a2, samples, "Hough lines, $dist^3$ weighting")

    samples = read_sample_file("./outputs/samples_hough_power10_movecam.dat")
    error_histogram_ax(a3, samples, "Hough lines, $dist^{10}$ weighting")

    samples = read_sample_file("./outputs/samples_contour_0_movecam.dat")
    error_histogram_ax(a4, samples, "Linear Regression")


    plt.tight_layout()
    plt.savefig("../report/plots/hough_power_comparison.pgf")


def power_comparision_plot2():
    fig, ((a1, a2, a3)) = plt.subplots(3, 1, sharex=True, sharey=True)

    samples = read_sample_file("./outputs/samples_hough_power0_stillcam.dat")
    error_histogram_ax(a1, samples, "No power weighting")

    samples = read_sample_file("./outputs/samples_hough_power3_stillcam.dat")
    error_histogram_ax(a2, samples, "$dist^3$ weighting")

    samples = read_sample_file("./outputs/samples_hough_power10_stillcam.dat")
    error_histogram_ax(a3, samples, "$dist^{10}$ weighting")


    plt.tight_layout()
    plt.savefig("../report/plots/hough_power_comparison_stillcam.pgf")


def exponent_plot(exp):
    x = np.linspace(0, 100, 100)
    y = np.power((1-(x/100)), exp)
    plt.plot(x, y, label=f"Exponent = {exp}")

def gen_exponent_plot():
    exponent_plot(0)
    exponent_plot(1)
    exponent_plot(3)
    exponent_plot(10)
    exponent_plot(20)
    plt.legend(loc="upper right")
    plt.xlabel("Distance from center (px)")
    plt.ylabel("Weight")
    plt.grid()
    plt.tight_layout()
    plt.savefig("../report/plots/exponent_weight.pgf")

#  gen_exponent_plot()


#  compute_all_samples()

#  samples = read_sample_file("./outputs/samples_hough_power0_stillcam.dat")
#  error_histogram(samples, "hough power 0")
#  samples = read_sample_file("./outputs/samples_hough_power3_stillcam.dat")
#  error_histogram(samples, "hough power 3")
#  samples = read_sample_file("./outputs/samples_hough_power10_stillcam.dat")
#  error_histogram(samples, "hough power 10")
#  samples = read_sample_file("./outputs/samples_contour_0_stillcam.dat")
#  error_histogram(samples, "contour")

#  samples = read_sample_file("./outputs/samples_hough_power0_movecam.dat")
#  error_histogram(samples, "hough power 0")
#  plt.savefig("../report/plots/hough_power_0_error.pgf")

#  power_comparision_plot()
#  plt.show()

samples = read_sample_file("./outputs/samples_hough_power0_stillcam.dat")
samples_within_1pc(samples)

samples = read_sample_file("./outputs/samples_hough_power3_stillcam.dat")
samples_within_1pc(samples)

samples = read_sample_file("./outputs/samples_hough_power10_stillcam.dat")
samples_within_1pc(samples)

samples = read_sample_file("./outputs/samples_contour_0_stillcam.dat")
samples_within_1pc(samples)

