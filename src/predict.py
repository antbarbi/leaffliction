import torch
import argparse
import matplotlib.pyplot as plt
from torch_classes import ImageDataset, CNN
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import json
from collections import OrderedDict


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


def plotting(img, prediction):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img)
    ax[0].axis('off')

    # ax[1].imshow()

    plt.figtext(0.5, 0.02, f"Class predicted: {prediction}", ha="center", fontsize=14, color="green")
    plt.suptitle("DL classification", fontsize=16, fontweight="bold")


def inspect_model(weights_path: str, config_path: str = "config.json"):
    """Extract model architecture and input shape information"""

    # Get input shape from config
    config_path = "./src/weights/shallow/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
        input_shape = tuple(config['input_size'])

    # Load state dict and get architecture
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)

    # Extract layer info
    architecture = OrderedDict()
    for name, param in state_dict.items():
        architecture[name] = {
            'shape': list(param.shape),
            'parameters': param.numel()
        }

    return {
        'input_shape': input_shape,
        'architecture': architecture,
        'total_parameters': sum(param.numel() for param in state_dict.values())
    }


def main(src: str, image_pth: str, weights_pth: str, map_location="cpu"):
    dataset = ImageDataset(src)
    print("Dataset is loaded for class labeling")

    res = inspect_model(weights_pth, map_location)
    print(res)
    exit()
    model = CNN(NUM_OF_CLASSES, dataset.resize)
    state_dict = torch.load(weights_pth, map_location, weights_only=True)
    layer = state_dict["conv_layer1.weight"]
    model.load_state_dict(state_dict)
    model.eval()

    img: torch.Tensor = read_image(image_pth, mode=ImageReadMode.RGB).float() / 255.0

    with torch.no_grad():
        input_tensor = img.unsqueeze(0)
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1)

    print(f"Prediction: {dataset.classes[prediction]}")

    plotting(img, dataset.classes[prediction])


if __name__ == "__main__":
    args = args_parser()
    print(args)
    main(args.source, args.image, args.weight)
