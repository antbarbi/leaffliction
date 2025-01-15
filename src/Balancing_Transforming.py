#!/usr/bin/env python3

import os
import argparse
import random
import Augmentation
import Transformation
from concurrent.futures import ProcessPoolExecutor


NUM_OF_AUGMENTATION = 6
Augmentation.verbose = False
Transformation.verbose = False


def process_directory(dir_path):
    Transformation.main(src=dir_path, dst=dir_path)


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Data augmentation",
    )
    parser.add_argument(
        "directory",
        help="The name of the folder where to do apply balancing",
        type=str
    )
    return parser.parse_args()


def main(directory: str):
    datasets = {}
    for root, dirs, files in os.walk(directory):
        if not dirs:
            datasets[len(files)] = (root, files)

    sorted_key = sorted(datasets.keys())
    MAX = sorted_key[-1]
    datasets = {key: datasets[key] for key in sorted_key[:-1]}

    for length, (path, filenames) in datasets.items():
        random.shuffle(filenames)
        for index, filename in enumerate(filenames):
            if length + index * NUM_OF_AUGMENTATION >= MAX:
                break
            pathname = os.path.join(path, filename)
            Augmentation.main(pathname)
    print("Datasets have been augmented and balanced.")

    with ProcessPoolExecutor() as executor:
        for root, dirs, files in os.walk(directory):
            dir_paths = [os.path.join(root, dir) for dir in dirs]
            executor.map(process_directory, dir_paths)
    print("Datasets have been transformed.")


if __name__ == "__main__":
    args = args_parser()
    main(args.directory)
