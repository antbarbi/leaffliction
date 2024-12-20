#!/usr/bin/env python3

import os
import argparse
from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import shutil


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
pcv.params.text_size = 20
pcv.params.text_thickness = 20


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
    colorspace_img = pcv.visualize.colorspaces(rgb_img=img)

    # Convert the image to grayscale
    b = pcv.rgb2gray_lab(rgb_img=img, channel="b")
    
    # Visualize the distribution of grayscales values
    hist = pcv.visualize.histogram(img=b)

    # Gaussian blur
    blur_img = pcv.gaussian_blur(img=img, ksize=(5, 5), sigma_x=0)

    # Convert to grayscale
    gray_img = pcv.rgb2gray_lab(rgb_img=img, channel='l')

    # Apply binary thresholding
    mask = pcv.threshold.binary(gray_img=gray_img, threshold=120, object_type='light')

    # Define ROI
    roi, roi_hierarchy = pcv.roi.rectangle(img=img, x=50, y=50, h=200, w=200)

    # Find objects
    id_objects, obj_hierarchy = pcv.find_objects(img=img, mask=mask)

    # Keep objects within the ROI
    roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img=img, roi_contour=roi, roi_hierarchy=roi_hierarchy, object_contour=id_objects, obj_hierarchy=obj_hierarchy, roi_type='partial')




if __name__ == "__main__":
    args = args_parser()
    main(args.src, args.dst, args.mask)
