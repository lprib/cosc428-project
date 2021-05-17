import cv2 as cv
import numpy as np

nothing = lambda x: None

def get_start_end(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    return (x1, y1), (x2, y2)

def main_hough():
    img_orig = cv.imread("single_knob4.png")
    img = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)

    cv.namedWindow('Hough Line Transform')
    cv.createTrackbar('CannyThreshold1', 'Hough Line Transform', 0, 1200, nothing)
    cv.createTrackbar('CannyThreshold2', 'Hough Line Transform', 0, 1200, nothing)
    cv.createTrackbar("HoughThreshold", 'Hough Line Transform', 0, 200, nothing)

    while True:
        cannyThreshold1 = cv.getTrackbarPos('CannyThreshold1', 'Hough Line Transform')
        cannyThreshold2 = cv.getTrackbarPos('CannyThreshold2', 'Hough Line Transform')
        houghThreshold = cv.getTrackbarPos('HoughThreshold', 'Hough Line Transform')

        drawing = img_orig.copy()
        edges = cv.Canny(img, cannyThreshold1, cannyThreshold2)
        lines = cv.HoughLines(edges, 1, np.pi/180, houghThreshold)

        #  print(lines.shape)
        if lines is not None:
            for line in lines:
                #  print(line)
                #  print()
                for rho,theta in line:
                    start, end = get_start_end(rho, theta)
                    cv.line(drawing, start, end, (0, 0, 255), 2)
            # Get list of just thetas
            #  thetas = lines[:,:,1].reshape((-1))
            # The lines are outputted in order of votes, so weight the first ones higher in avg
            #  avg_theta = np.average(thetas)
            #  avg_theta = np.average(thetas, weights=np.flip(np.arange(len(thetas))))
            # Due to the format of (rho, theta), need to rotate 90 to get angle around origin
            #  avg_theta = avg_theta - np.pi / 2
            avg_theta = thetas[0] - np.pi / 2


            # draw line from center of image
            center = (int(drawing.shape[0] / 2), int(drawing.shape[1] / 2))
            r = center[0]
            dims = (int(r*np.sin(avg_theta)), int(r*np.cos(avg_theta)))
            cv.line(drawing, center, (center[0] + dims[0], center[1] + dims[1]), (255, 0, 0), 3)


        combined = np.concatenate((drawing, cv.cvtColor(edges, cv.COLOR_GRAY2BGR)), axis=1)
        cv.imshow('Hough Line Transform', combined)

        if cv.waitKey(1000) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break

def main_erosion():
    img = cv.imread("single_knob.png")
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.namedWindow('erosion angle')
    cv.createTrackbar('adaptive C', 'erosion angle', 0, 255, nothing)

    while True:
        thresh_c = cv.getTrackbarPos('adaptive C', 'erosion angle')
        blurred = cv.GaussianBlur(img_gray, (5, 5), 0)
        ret, thresh = cv.threshold(blurred, thresh_c, 255, cv.THRESH_BINARY)
        combined = np.concatenate((img_gray, thresh), axis=1)
        cv.imshow('erosion angle', combined)
        if cv.waitKey(1000) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break

def edge_morph(filename, cannyThreshold1, cannyThreshold2):
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    morph_kernel = np.ones((3, 3), np.uint8)

    edges = cv.Canny(img, cannyThreshold1, cannyThreshold2)
    morphed = cv.morphologyEx(edges, cv.MORPH_CLOSE, morph_kernel, iterations=2)

    contours, _hierarchy = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv.contourArea(x))
    box = img.copy()
    if len(contours) > 0:
        largest = contours[0]
        [vx, vy, x, y] = cv.fitLine(largest, cv.DIST_L2, 0, 0.1, 0.1)
        cols = img.shape[1]
        l = int((-x*vy/vx) + y)
        r = int(((cols-x)*vy/vx)+y)
        cv.line(box,(cols-1,r),(0,l),(255),2)

    combined = np.concatenate((img, edges, morphed, box), axis=1)
    return combined

def main_edge_morph():

    cv.namedWindow('edge morph')
    cv.createTrackbar('CannyThreshold1', 'edge morph', 59, 255, nothing)
    cv.createTrackbar('CannyThreshold2', 'edge morph', 144, 255, nothing)

    morph_kernel = np.ones((3, 3), np.uint8)

    while True:
        cannyThreshold1 = cv.getTrackbarPos('CannyThreshold1', 'edge morph')
        cannyThreshold2 = cv.getTrackbarPos('CannyThreshold2', 'edge morph')
        combined = np.concatenate((
            edge_morph("single_knob000.png", cannyThreshold1, cannyThreshold2),
            edge_morph("single_knob001.png", cannyThreshold1, cannyThreshold2),
            edge_morph("single_knob002.png", cannyThreshold1, cannyThreshold2),
            edge_morph("single_knob003.png", cannyThreshold1, cannyThreshold2)),
            axis=0
        )
        cv.imshow('edge morph', combined)

        if cv.waitKey(1000) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break


#  main_hough()
#  main_erosion()
main_edge_morph()
