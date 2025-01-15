import torch
import argparse
import matplotlib.pyplot as plt
from torch_classes import ImageDataset, CNN
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms


NUM_OF_CLASSES = 8

def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Data augmentation",
    )
    parser.add_argument(
        "-s",
        "--source",
        help="The src has to be a folder \
            behavior differs depending on the type of the src",
        type=str
    )
    parser.add_argument(
        "-i",
        "--image",
        help="The image to have to prediction on",
        type=str
    )
    parser.add_argument(
        "-w",
        "--weight",
        help="The image to have to prediction on",
        type=str
    )
    return parser.parse_args()


def main(src: str, image_pth: str, weights_pth: str, map_location="cpu"):
    dataset = ImageDataset(src)
    print("Dataset is loaded for class labeling")

    model = CNN(NUM_OF_CLASSES, dataset.resize)
    model.load_state_dict(torch.load(weights_pth, map_location, weights_only=True))
    model.eval()

    img: torch.Tensor = read_image(image_pth, mode=ImageReadMode.RGB).float() / 255.0

    with torch.no_grad():
        input_tensor = img.unsqueeze(0)
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1)

    print(f"Prediction: {dataset.classes[prediction]}")


if __name__ == "__main__":
    args = args_parser()
    print(args)
    main(args.source, args.image, args.weight)
