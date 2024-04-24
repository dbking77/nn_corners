#!/usr/bin/env python3

import argparse
from collections import defaultdict
import csv
import cv2 as cv
import os


def main():
    parser = argparse.ArgumentParser(description="Extracts images based on annotated point features")
    parser.add_argument("csv", help="CSV file with annoated point information")
    parser.add_argument("imgdir", help="Directory with input images")
    parser.add_argument("outdir", help="Output directory for extracted sub-images")
    parser.add_argument("--size", type=int, default=300,
                        help="Size in pixels of rectangle to extract around center pixel")
    parser.add_argument("--scale", type=float, default=0.3,
                        help="scale image before extracting sub-rectangle")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="ratio of images for each label to reserve for validation set")
    parser.add_argument("--val-min", type=int, default=2,
                        help="keep at least these many images for validation set even if ratio would produce less")
    args = parser.parse_args()

    labels = defaultdict(list)
    with open(args.csv, 'r') as fd:
        reader = csv.reader(fd, delimiter=',')
        for (label, x, y, img_fn, *_) in reader:
            labels[label].append((int(x), int(y), img_fn))

    sz1 = args.size//2
    sz2 = args.size - sz1
    for label, label_data in labels.items():
        print(f"{label} {len(label_data)}")
        subdirs = {}
        for name in ('val', 'train'):
            subdirs[name] = os.path.join(args.outdir, name, label)
            os.makedirs(subdirs[name], exist_ok=True)

        val_count = max(len(label_data) * args.val_ratio, args.val_min)
        if val_count >= len(label_data):
            print(f"WARN : not enough data for {label} for validation and training")

        for idx, (x, y, img_fn) in enumerate(label_data):
            fn = os.path.join(args.imgdir, img_fn)
            img = cv.imread(fn)
            if img is None:
                print("Cannot load image", img_fn)
                continue
            img = cv.resize(img, None, fx=args.scale, fy=args.scale)
            x, y = (int(round(x*args.scale)), int(round(y*args.scale)))
            img = img[y - sz1: y + sz2, x - sz1: x + sz2]
            subdir = subdirs['val' if (idx < val_count) else 'train']
            outfn = os.path.join(subdir, f"{label}_{idx:03d}.jpg")
            cv.imwrite(outfn, img)

if __name__ == "__main__":
    main()