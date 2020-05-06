#!/usr/bin/env

import sys, os
import cv2 as cv
import numpy as np

def compute_histograms(class_name, instance_files):
    color_histograms = []
    circle_histograms = [] # histogram of circle sizes in the image
    for f in instance_files:
        image = cv.imread(f)
        color_hist = cv.calcHist([image], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256])
        cv.normalize(color_hist, color_hist, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
        color_histograms.append(color_hist)

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray_blur = cv.blur(gray, (3, 3))
        circle_sizes = []
        # We get different results bounding the radius versus just giving the full range at once.
        for circle_size in range(3, 15+1):
            circles = cv.HoughCircles(gray_blur, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=20, minRadius=circle_size, maxRadius=circle_size)
            if circles is not None:
                for c in circles[0]:
                    circle_sizes.append((c[2],))
        if circle_sizes:
            circle_sizes = np.array(circle_sizes, dtype=np.float32)
            circle_hist = cv.calcHist([circle_sizes], [0], None, [13], [3, 16])
            cv.normalize(circle_hist, circle_hist, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
        else:
            circle_hist = np.zeros((13, 1), dtype=np.float32)
        circle_histograms.append(circle_hist)

    color_cross_chi = [cv.compareHist(h, j, cv.HISTCMP_CHISQR) for h in color_histograms for j in color_histograms if id(h) != id(j)]
    if len(color_cross_chi):
        color_avg_chi = sum(color_cross_chi)/len(color_cross_chi)
    else:
        color_avg_chi = -1.0

    circle_cross_chi = [cv.compareHist(h, j, cv.HISTCMP_CHISQR) for h in circle_histograms for j in circle_histograms if id(h) != id(j)]
    if len(circle_cross_chi):
        circle_avg_chi = sum(circle_cross_chi)/len(circle_cross_chi)
    else:
        circle_avg_chi = -1.0

    print('{:20} {:6.2f} {:6.2f}'.format(class_name, color_avg_chi, circle_avg_chi))

    # We average the histograms without renormalizing them, particularly so that
    # an all-0 circle histogram is weighted against histograms with some 1s.
    color_hist = sum(color_histograms)/len(color_histograms)
    circle_hist = sum(circle_histograms)/len(circle_histograms)
    return color_hist, circle_hist

def main(args):
    class_dirs = os.listdir('data/classes/')
    class_dirs.sort()
    class_to_instances = {}
    for cd in class_dirs:
        instances = os.listdir('data/classes/'+cd)
        instances.sort()
        class_to_instances[cd] = ['data/classes/'+cd+'/'+instance for instance in instances]
    class_to_hists = {}
    for c, instances in class_to_instances.items():
        class_to_hists[c] = compute_histograms(c, instances)
    for c, hs in class_to_hists.items():
        for d, js in class_to_hists.items():
            chi = [cv.compareHist(h, j, cv.HISTCMP_CHISQR) for (h, j) in zip(hs, js)]
            chi_str = ' '.join('{:6.2f}'.format(c) for c in chi)
            print('{:20} {:20} {}'.format(c, d, chi_str))

if __name__ == '__main__':
    main(None)