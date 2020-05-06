#!/usr/bin/env

import sys, os
import cv2 as cv

def compute_histograms(class_name, instance_files):
    color_histograms = []
    circle_histograms = [] # histogram of circle sizes in the image
    for f in instance_files:
        image = cv.imread(f)
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        hist = cv.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv.normalize(hist, hist, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
        color_histograms.append(hist)

    color_cross_chi = [cv.compareHist(h, j, cv.HISTCMP_CHISQR) for h in color_histograms for j in color_histograms if id(h) != id(j)]
    if len(color_cross_chi):
        color_avg_chi = sum(color_cross_chi)/len(color_cross_chi)
    else:
        color_avg_chi = 0.0

    print('{:20} {:6.2f}'.format(class_name, color_avg_chi))

    color_hist = sum(color_histograms)
    cv.normalize(color_hist, color_hist, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
    return (color_hist,)

def main(args):
    class_dirs = os.listdir('data/classes/')
    class_dirs.sort()
    class_to_instances = {}
    for cd in class_dirs:
        instances = os.listdir('data/classes/'+cd)
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