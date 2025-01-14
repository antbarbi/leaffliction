import torch
import argparse
import os
from torch_classes import ImageDataset, CNN

NUM_OF_CLASSES = 8

def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Data augmentation",
    )
    parser.add_argument(
        "src",
        help="The src has to be a folder \
            behavior differs depending on the type of the src",
        type=str
    )
    parser.add_argument(
        "img",
        help="The image to have to prediction on",
        type=str
    )
    parser.add_argument(
        "weight",
        help="The image to have to prediction on",
        type=str
    )
    return parser.parse_args()


def main(src: str, image_pth: str, weights_pth: str, map_location="cpu"):
    print("Loading dataset")
    dataset = ImageDataset(src)
    print("Dataset is loaded")

    model = CNN(NUM_OF_CLASSES, dataset.resize)
    model.load_state_dict(torch.load(weights_pth, map_location))
    model.eval()

    img: torch.Tensor = dataset[0][0]
    print(img.shape, dataset[0][1], dataset.classes[dataset[0][1]])

    # with torch.no_grad():
    #     input_tensor = img
    #     output = model(input_tensor)
    #     prediction = torch.argmax(output, dim=1)
    #     print(prediction)


if __name__ == "__main__":
    args = args_parser()
    print(args)
    main(args.src, args.img, args.weight)
