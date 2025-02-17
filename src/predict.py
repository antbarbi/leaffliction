import torch
import argparse
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


INDEX = 91000

def main(src: str, image_pth: str, weights_pth: str, map_location="cpu"):
    print("Loading dataset")
    dataset = ImageDataset(src, (3, 128, 128))
    print("Dataset is loaded")

    model = CNN(NUM_OF_CLASSES, dataset.resize)
    model.load_state_dict(torch.load(weights_pth, map_location, weights_only=True))
    model.eval()

    img: torch.Tensor = dataset[INDEX][0]
    print(img.shape, dataset[INDEX][1], dataset.classes[dataset[INDEX][1]])

    with torch.no_grad():
        input_tensor = img.unsqueeze(0)
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1)

    print(f"Prediction: {dataset.classes[prediction]}")


if __name__ == "__main__":
    args = args_parser()
    print(args)
    main(args.src, args.img, args.weight)
