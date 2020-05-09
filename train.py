#!/usr/bin/env
import shutil
import sys, os
import math
from math import atan2, sqrt
from pathlib import Path
from typing import Dict, List
import cv2 as cv
import numpy as np
from PIL import Image


class Classifier:
    def __init__(self, class_to_instances: Dict[str, List[Path]]):
        # Ensure we only make each mistake once by retaining the hashes of our
        # training set.  (It's safe to have non-hash filenames; they just won't
        # be matched.)
        self.instance_to_class = {}
        self.class_to_histograms = {}
        for c, instances in class_to_instances.items():
            for i in instances:
                self.instance_to_class[i.stem] = c
            self.class_to_histograms[c] = Classifier._compute_histograms_for_class(c, instances)

        # for c, hs in self.class_to_histograms.items():
        #     for d, js in self.class_to_histograms.items():
        #         chi = [cv.compareHist(h, j, cv.HISTCMP_CHISQR_ALT) for (h, j) in zip(hs, js)]
        #         chi_str = ' '.join('{:6.2f}'.format(c) for c in chi)
        #         print('{:20} {:20} {}'.format(c, d, chi_str))

    def __call__(self, image, pixel_data_hash=None):
        return self.classify(image, pixel_data_hash)[0]

    def classify(self, image, pixel_data_hash=None):
        if isinstance(image, str):
            image = cv.imread(image)
        if isinstance(image, Path):
            image = cv.imread(str(image))
        if isinstance(image, Image.Image):
            # https://stackoverflow.com/a/14140796
            image = np.array(image)
            cv.cvtColor(image, cv.COLOR_RGB2BGR)

        if pixel_data_hash:
            c = self.instance_to_class.get(pixel_data_hash)
            if c:
                return c, 0.0

        ihs = Classifier._compute_histograms_for_image(image)
        best_score = math.inf
        best_class = None
        for name, chs in self.class_to_histograms.items():
            score = sum(cv.compareHist(ih, ch, cv.HISTCMP_CHISQR_ALT) for (ih, ch) in zip(ihs, chs))
            if score < best_score:
                best_score = score
                best_class = name
        return best_class, best_score

    @staticmethod
    def _compute_keypoint_histogram(image, detector):
        kps = detector.detect(image)
        polar_kps = []
        for kp in kps:
            x, y = kp.pt
            dist = sqrt((x - 31) ** 2 + (y - 31) ** 2)
            if dist >= 20: continue
            phi = atan2(y, x)
            polar_kps.append([(dist, phi)])
        if polar_kps:
            polar_kps = np.array(polar_kps, dtype=np.float32)
            kp_hist = cv.calcHist([polar_kps], [0, 1], None, [20, 16], [0, 20, -math.pi, math.pi])
            cv.normalize(kp_hist, kp_hist, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
        else:
            kp_hist = np.ones((20, 16), dtype=np.float32) / (20*16) # unprincipled ones
        return kp_hist

    @staticmethod
    def _compute_histograms_for_image(image):
        color_hist = cv.calcHist([image], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
        cv.normalize(color_hist, color_hist, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray_blur = cv.blur(gray, (3, 3))
        circle_sizes = []
        # We get different results bounding the radius versus just giving the full range at once.
        for circle_size in range(3, 15 + 1):
            circles = cv.HoughCircles(gray_blur, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=20, minRadius=circle_size, maxRadius=circle_size)
            if circles is not None:
                for c in circles[0]:
                    circle_sizes.append((c[2],))
        if circle_sizes:
            circle_sizes = np.array(circle_sizes, dtype=np.float32)
            circle_hist = cv.calcHist([circle_sizes], [0], None, [13], [3, 16])
            cv.normalize(circle_hist, circle_hist, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
        else:
            circle_hist = np.ones((13, 1), dtype=np.float32) / 13 # unprincipled ones

        gftt_hist = Classifier._compute_keypoint_histogram(image, cv.GFTTDetector.create(maxCorners=10000))
        mser_hist = Classifier._compute_keypoint_histogram(image, cv.MSER.create())
        return color_hist, circle_hist, gftt_hist, mser_hist

    @staticmethod
    def _compute_histograms_for_class(class_name, instance_files):
        color_histograms = []
        circle_histograms = []  # histogram of circle sizes in the image
        gftt_histograms = []
        mser_histograms = []
        for f in instance_files:
            if isinstance(f, Path):
                f = str(f)
            image = cv.imread(f)
            color_hist, circle_hist, gftt_hist, mser_hist = Classifier._compute_histograms_for_image(image)
            color_histograms.append(color_hist)
            circle_histograms.append(circle_hist)
            gftt_histograms.append(gftt_hist)
            mser_histograms.append(mser_hist)

        # color_avg_chi = avg_chi(color_histograms)
        # circle_avg_chi = avg_chi(circle_histograms)
        # gftt_avg_chi = avg_chi(gftt_histograms)
        # mser_avg_chi = avg_chi(mser_histograms)
        # print('{:20} {:6.2f} {:6.2f} {:6.2f} {:6.2f}'.format(class_name, color_avg_chi, circle_avg_chi, gftt_avg_chi, mser_avg_chi))

        # We average the histograms without renormalizing them, particularly so that
        # an all-0 circle histogram is weighted against histograms with some 1s.
        color_hist = sum(color_histograms) / len(color_histograms)
        circle_hist = sum(circle_histograms) / len(circle_histograms)
        gftt_hist = sum(gftt_histograms) / len(gftt_histograms)
        mser_hist = sum(mser_histograms) / len(mser_histograms)
        return color_hist, circle_hist, gftt_hist, mser_hist

    @staticmethod
    def _avg_chi(histograms):
        cross_chi = [cv.compareHist(h, j, cv.HISTCMP_CHISQR_ALT) for h in histograms for j in histograms if
            id(h) != id(j)]
        if len(cross_chi):
            avg_chi = sum(cross_chi) / len(cross_chi)
        else:
            avg_chi = -1.0
        return avg_chi


    def self_test(self, class_to_instances):
        """Prints the members of the training set that are misclassified against the class histograms."""
        for c, instances in class_to_instances.items():
            for i in instances:
                best_class, best_score = self.classify(i)
                correct_score = sum(cv.compareHist(ih, ch, cv.HISTCMP_CHISQR_ALT) for (ih, ch) in zip(self._compute_histograms_for_image(cv.imread(str(i))), self.class_to_histograms[c]))
                if best_class != c:
                    print(i, best_class, best_score, correct_score)

def main(args):
    class_dirs = [x for x in Path('data/classes').iterdir()]
    class_dirs.sort()
    class_to_instances = {}
    for cd in class_dirs:
        instances = [x for x in cd.iterdir()]
        instances.sort()
        class_to_instances[cd.name] = instances
    classifier = Classifier(class_to_instances)
    classifier.self_test(class_to_instances)

    # input_files = ['/tmp/output2/'+x for x in os.listdir('/tmp/output2')]
    # input_files.sort()
    # for c in class_to_instances:
    #     try:
    #         os.mkdir('/tmp/output2/'+c)
    #     except FileExistsError:
    #         pass
    # for i in input_files:
    #     c, _ = classify(class_to_hists, i)
    #     shutil.move(i, '/tmp/output2/'+c)


if __name__ == '__main__':
    main(None)