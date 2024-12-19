#!/usr/bin/env python3

import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Data augmentation",
    )
    parser.add_argument(
        "image_name",
        help="The name of the image on which to do augmentation",
        type=str
    )
    parser.add_argument(
        "-v", "--visual",
        action='store_true',
        default=False,
        help="Enable visualization"
    )
    return parser.parse_args()


def visualizer(augmented: dict) -> None:
    num_of_images = len(augmented)
    if num_of_images // 2 != num_of_images / 2:
        fig, axes = plt.subplots(2, num_of_images // 2 + 1, figsize=(12, 8))
        axes[-1, -1].axis('off')
    else:
        fig, axes = plt.subplots(2, num_of_images // 2, figsize=(12, 8))
    print(2, num_of_images // 2)

    for ax, (transf, img) in zip(axes.flatten(), augmented.items()):
        ax.imshow(img)
        ax.set_title(transf)
        ax.axis('off')

    plt.show()


def main(image_name: str, visual: bool = False):
    if not os.path.exists(image_name):
        raise FileNotFoundError(
            f"{image_name} not found or doesn't exists."
        )

    image = Image.open(image_name)

    flipper = transforms.RandomHorizontalFlip(p=1)
    rotater = transforms.RandomRotation(degrees=90)
    skewer = transforms.RandomAffine(degrees=0, shear=60)
    cropper = transforms.RandomResizedCrop(size=image.size)
    distorter = transforms.RandomPerspective(distortion_scale=0.5, p=1)
    blurer = transforms.GaussianBlur(kernel_size=5, sigma=2.0)

    augmented = {
        "Flip": flipper(image),
        "Rotate": rotater(image),
        "Skew": skewer(image),
        "Crop": cropper(image),
        "Distortion": distorter(image),
        "Blur": blurer(image)
    }

    for transf, img in augmented.items():
        name, ext = os.path.splitext(image_name)
        print(f"{name}_{transf}{ext}")
        image.save()

    augmented["Normal"] = image
    if visual:
        visualizer(augmented)
    

if __name__ == "__main__":
    args = args_parser()
    main(args.image_name, args.visual)
