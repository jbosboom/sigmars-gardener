#!/usr/bin/env python3

from math import sqrt
import time
import hashlib
import pickle
from pathlib import Path
import pyautogui
import xdg
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

def parse_screenshot(screenshot, classifier):
    puzzle = {}
    for i, region_center in enumerate(data.centers):
        image, pixel_data_hash = extract_and_clean(screenshot, region_center)
        thing = classifier(image, pixel_data_hash)
        if thing.endswith('-inactive'):
            thing = thing[:-9]
        if thing != 'blank':
            puzzle[i] = thing
    return puzzle

def main(args):
    classifier_hash = hashlib.sha1()
    class_dirs = [x for x in Path('data/classes').iterdir()]
    class_dirs.sort()
    class_to_instances = {}
    for cd in class_dirs:
        instances = [x for x in cd.iterdir()]
        instances.sort()
        class_to_instances[cd.name] = instances
        classifier_hash.update(cd.name.encode())
        for i in instances:
            classifier_hash.update(str(i).encode())
    classifier_hash = classifier_hash.hexdigest()
    classifier_pickle_file = xdg.XDG_CACHE_HOME / ('sigmars-gardener-classifier-' + classifier_hash)
    try:
        with open(classifier_pickle_file, 'rb') as f:
            classifier = pickle.load(f)
    except FileNotFoundError:
        classifier = train.Classifier(class_to_instances)
        with open(classifier_pickle_file, 'wb') as f:
            pickle.dump(classifier, f, protocol=pickle.HIGHEST_PROTOCOL)

    upper_left_x, upper_left_y, _, _ = pyautogui.locateOnScreen('data/upper-left.png')
    screenshot = pyautogui.screenshot(region=(upper_left_x, upper_left_y, 1920, 1080))
    puzzle = parse_screenshot(screenshot, classifier)
    print(puzzle)

    histogram = {v: 0 for v in puzzle.values()}
    for v in puzzle.values():
        histogram[v] += 1
    histogram['blank'] = data.dummy - len(puzzle)
    expected = {
        'air': 8, 'earth': 8, 'fire': 8, 'water': 8, 'salt': 4,
        'quicksilver': 5, 'lead': 1, 'tin': 1, 'iron': 1, 'copper': 1, 'silver': 1, 'gold': 1,
        'vitae': 4, 'mors': 4,
        'blank': 36,
    }
    for k, v in histogram.items():
        if expected[k] != v:
            print('wrong number of {}: expected {}, found {}'.format(k, expected[k], v))

    moves = solver.solve(puzzle)
    print(moves)

    while moves:
        a, b = moves.pop(0) # could reverse the list, I guess
        cx, cy = data.centers[a]
        pyautogui.moveTo(upper_left_x + cx, upper_left_y + cy, duration=0.1)
        time.sleep(0.1)
        pyautogui.mouseDown()
        time.sleep(0.1)
        pyautogui.mouseUp()
        cx, cy = data.centers[b]
        pyautogui.moveTo(upper_left_x + cx, upper_left_y + cy, duration=0.1)
        time.sleep(0.1)
        pyautogui.mouseDown()
        time.sleep(0.1)
        pyautogui.mouseUp()
        pyautogui.moveTo(upper_left_x, upper_left_y, duration=0.1)
        time.sleep(0.1)

        # Check if our identification of any puzzle elements changed.  If so,
        # write out the old screenshot and re-solve with the new information.
        # We might fail to re-solve if we've used up elements we need.
        mistaken = False
        new_screenshot = pyautogui.screenshot(region=(upper_left_x, upper_left_y, 1920, 1080))
        new_puzzle = parse_screenshot(new_screenshot, classifier)
        for k, v in new_puzzle.items():
            if puzzle[k] != v:
                mistake_dir = xdg.XDG_RUNTIME_DIR / 'sigmars-gardener'
                mistake_dir.mkdir(exist_ok=True)
                mistake_image, mistake_hash = extract_and_clean(screenshot, data.centers[k])
                mistake_image.save(mistake_dir / (mistake_hash + '.png'))
                mistaken = True
        if mistaken:
            moves = solver.solve(new_puzzle)
            if not moves:
                raise Exception("mistake made puzzle unsolvable")


if __name__ == '__main__':
    main(None)