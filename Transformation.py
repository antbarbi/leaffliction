#!/usr/bin/env python3

import os
import argparse
import shutil
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs


# # Input/output options
# args = WorkflowInputs(
#     images=["./img/arabidopsis.jpg"],
#     names="image1",
#     result="segmentation_results.json",
#     outdir=".",
#     writeimg=False,
#     debug="plot"
#     )

if os.path.exists("./debug_outputs"):
    shutil.rmtree("./debug_outputs")
os.mkdir("./debug_outputs")

pcv.params.debug = "print"
pcv.params.debug_outdir = "./debug_outputs"
pcv.params.dpi = 100
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
        type=str
    )
    parser.add_argument(
        "-mask",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main(src: str, dst: str, mask: bool = False):
    if not os.path.exists(src):
        raise FileNotFoundError("File or folder not found or doesn't exists.")

    img, path, filename = pcv.readimage(filename=src)

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


    # Region of interest
    roi = pcv.roi.from_binary_image(img, b_filt)

    # Connecting or splitting object
    shape = pcv.analyze.size(img=img, labeled_mask=b_filt, n_labels=1)


if __name__ == "__main__":
    args = args_parser()
    main(args.src, args.dst, args.mask)
