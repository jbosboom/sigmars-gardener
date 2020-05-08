#!/usr/bin/env python3

from math import sqrt
import hashlib
from pathlib import Path
import pyautogui
import data, train, solver

def extract_and_clean(screenshot, region_center):
    cx, cy = region_center
    pixel_data = []
    region = screenshot.crop((cx - 30, cy - 30, cx + 31, cy + 31))
    for y in range(cy - 30, cy + 30 + 1):
        for x in range(cx - 30, cx + 30 + 1):
            dist = sqrt((x - cx) ** 2 + (y - cy) ** 2)
            newpoint = (x - (cx - 30), y - (cy - 30))
            if dist > 30:
                region.putpixel(newpoint, (255, 255, 255))
            else:
                r, g, b = region.getpixel(newpoint)
                pixel_data.extend((r, g, b))
    digester = hashlib.sha1()
    digester.update(bytes(pixel_data))
    pixel_data_hash = digester.hexdigest()
    return region, pixel_data_hash

def main(args):
    class_dirs = [x for x in Path('data/classes').iterdir()]
    class_dirs.sort()
    class_to_instances = {}
    for cd in class_dirs:
        instances = [x for x in cd.iterdir()]
        instances.sort()
        class_to_instances[cd.name] = instances
    classifier = train.Classifier(class_to_instances)

    upper_left_x, upper_left_y, _, _ = pyautogui.locateOnScreen('data/upper-left.png')
    screenshot = pyautogui.screenshot(region=(upper_left_x, upper_left_y, 1920, 1080))
    puzzle = {}
    for i, region_center in enumerate(data.centers):
        image, pixel_data_hash = extract_and_clean(screenshot, region_center)
        thing = classifier(image, pixel_data_hash)
        if thing.endswith('-inactive'):
            thing = thing[:-9]
        if thing != 'blank':
            puzzle[i] = thing
    print(puzzle)
    moves = solver.solve(puzzle)
    print(moves)


if __name__ == '__main__':
    main(None)