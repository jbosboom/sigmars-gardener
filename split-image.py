#!/usr/bin/env python3

# Extracts grid spaces to separate files for human sorting.

import sys
import hashlib
from math import sqrt
from PIL import Image

import data

ctr = 0
for image_filename in sys.argv[1:]:
    with Image.open(image_filename) as im:
        for cx, cy in data.centers:
            pixel_data = []
            region = im.crop((cx - 30, cy - 30, cx + 31, cy + 31))
            for y in range(cy-30, cy+30+1):
                for x in range(cx-30, cx+30+1):
                    dist = sqrt((x - cx)**2 + (y - cy)**2)
                    newpoint = (x-(cx-30), y-(cy-30))
                    if dist > 30:
                        region.putpixel(newpoint, (255, 255, 255))
                    else:
                        r, g, b = region.getpixel(newpoint)
                        pixel_data.extend((r, g, b))
            digester = hashlib.sha1()
            digester.update(bytes(pixel_data))
            pixel_data_hash = digester.hexdigest()
            region.save("/tmp/output/{}.png".format(pixel_data_hash))
            ctr += 1