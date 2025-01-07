#!/usr/bin/env python3

import os
import argparse
import shutil
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import cv2
import numpy as np

# if os.path.exists("./debug_outputs"):
#     shutil.rmtree("./debug_outputs")
# os.mkdir("./debug_outputs")

# pcv.params.debug = "print"
# pcv.params.debug_outdir = "./debug_outputs"
# pcv.params.dpi = 100
# pcv.params.text_size = 20
# pcv.params.text_thickness = 20


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Data augmentation",
    )
    parser.add_argument(
        "-src",
        help="The src can be a folder or an image, behavior differs depending on the type of the src",
        type=str
    )
    parser.add_argument(
        "-dst",
        help="Folder destination",
        type=str,
        default=None,
        required=False
    )
    return parser.parse_args()


def roi_image(img, roi):
    new_img = np.copy(img)

    contours = roi.contours[0]
    hierarchy = roi.hierarchy[0]

    # Draw all contours in green
    cv2.drawContours(new_img, contours, -1, (0, 255, 0), 2)
    return new_img


def display_images(*args: tuple, **kwargs: dict) -> None:
    num_of_images = int(len(kwargs))

    if num_of_images > 1:
        if num_of_images % 2 == 0:
            fig, axes = plt.subplots(2, num_of_images // 2, figsize=(8,6))
        else:
            fig, axes = plt.subplots(2, num_of_images // 2 + 1, figsize=(8,6))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 1)
        axes = [axes]

    for (name, img), axe in zip(kwargs.items(), axes):
        try:
            if name.lower() in ['roi_image', 'thresholded_image', 'filtered_image', 'grayscale_image']:
                axe.imshow(img, cmap='gray')
            else:
                axe.imshow(img)
            axe.set_title(name)
            axe.axis('off')
        except Exception as e:
            print(f"Error displaying {name}: {e}")
            axe.axis('off')  # Hide the axis if there's an error
            continue

    axes[-1].axis('off')
    plt.tight_layout()
    plt.show()


def tranformations(filename: str, show: bool=False) -> None:
    img, path, filename = pcv.readimage(filename=filename)

    # Visualize colorspaces HSV / LAB
    colorspace_img = pcv.visualize.colorspaces(rgb_img=img, original_img=False)

    # Convert the image to grayscale
    b = pcv.rgb2gray_lab(rgb_img=img, channel="a")

    ## Isolating the image from background
    # Threshold the image
    b_thresh = pcv.threshold.otsu(gray_img=b, object_type='dark')
    # Noise reduction
    b_filt = pcv.median_blur(gray_img=b_thresh, ksize=5)

    ## Morphology Analysis
    skeleton = pcv.morphology.skeletonize(mask=b_filt)
    pruned_skel, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, size=50, mask=b_filt)
    prim_seg, second_seg = pcv.morphology.segment_sort(skel_img=pruned_skel, objects=edge_objects, mask=b_filt)
    filled_img = pcv.morphology.fill_segments(mask=b_filt, objects=prim_seg, stem_objects=second_seg, label="default")

    # Region of interest
    roi = pcv.roi.from_binary_image(img, b_filt)
    roi_img = roi_image(img, roi)

    # Connecting or splitting object
    shape = pcv.analyze.size(img=img, labeled_mask=b_filt, n_labels=1)

    filtered_img = pcv.apply_mask(img=img, mask=b_filt, mask_color="white")

    if show:
        display_images(
            Original_Image=img,
            Grayscale_Image=b,
            Filtered_Image=b_filt,
            Skeletonized_Image=skeleton,
            ROI=roi_img,
            Filled_Segments=filled_img,
            Masked_Image=filtered_img  # Ensure all passed arguments are images
            # Removed 'shape' as it is not an image
        )


def main(src: str, dst: str = None):
    if not os.path.exists(src):
        raise FileNotFoundError("File or folder not found or doesn't exists.")

    if os.path.isfile(src):
        tranformations(src, show=True)
    if os.path.isdir(src):
        pass


if __name__ == "__main__":
    args = args_parser()
    main(args.src, args.dst)
