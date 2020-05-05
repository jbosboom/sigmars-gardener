#!/usr/bin/env python3

import sys
from operator import itemgetter
from typing import Tuple, List
from math import atan2, sqrt, pi

import cv2 as cv
import numpy as np

# Input: a blank Sigmar's Garden board (as after solving) with other parts of
# the image, including the decorations in the corners around the board, replaced
# with a solid color.
img = cv.imread(sys.argv[1])
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
kp = cv.MSER.create().detect(img_gray, None)

img_blur = cv.blur(img_gray, (3, 3))
circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, 1, minDist=20, param1=50, param2=20, minRadius=30, maxRadius=35)
circles = np.uint16(np.around(circles))

# For each circle center we want the closest MSER point.  (The circles vary
# slightly in roundness; MSER gets the region center right but has stray points.)
points = []
for i in circles[0,:]:
    x, y = i[0], i[1]
    best, best_dist = None, float('inf')
    for k in kp:
        x2, y2 = k.pt
        dist = (x - x2)**2 + (y - y2)**2
        if dist < best_dist:
            best = k
            best_dist = dist
    points.append(best)

drawing = img.copy()
cv.drawKeypoints(img, points, drawing, color=(0, 255, 0), flags=0)
cv.imwrite('/tmp/find-grid.png', drawing)

# These don't all round the same way, but I'm betting it doesn't matter enough.
points = [(round(p.pt[0]), round(p.pt[1])) for p in points]
points.sort(key=lambda p: (p[1], p[0]))
print("centers =", tuple(points))

dummy = (0, 0)
pure_directions = [0.0, -1.059, -2.109, 3.142, 2.109, 1.059]
direction_intervals = [(x-0.1, x+0.1) for x in pure_directions]
def adj_atan2(y, x):
    phi = atan2(y, x)
    # if close to -pi, add 2*pi so we don't get confused by the discontinuity from pi to -pi
    if phi < -pi + 0.1:
        phi += 2*pi
    return phi
# For each point, convert all the other points to polar coordinates, then take
# the closest (least-distance) point in each of the six directions, or the dummy
# point if there aren't any points in that direction.
neighbor_points: List[List[Tuple[int, int]]] = []
for p in points:
    x, y = p
    polar = {(sqrt((x - x2)**2 + (y - y2)**2), adj_atan2(y2 - y, x2 - x)): (x2, y2)
            for (x2, y2) in points if x != x2 or y != y2}
    neighbors = []
    for di in direction_intervals:
        dp = [p for p in polar if di[0] <= p[1] <= di[1]]
        dp.sort(key=itemgetter(0))
        neighbors.append(polar[dp[0]] if dp else dummy)
    neighbor_points.append(neighbors)

point_to_index = {p: i for (i, p) in enumerate(points)}
point_to_index[dummy] = len(point_to_index)
neighbor_points = [[point_to_index[x] for x in l] for l in neighbor_points]
print("neighbors = (")
for l in neighbor_points:
    print("    ", tuple(l), ",", sep='')
print(")")
print("dummy = 91")