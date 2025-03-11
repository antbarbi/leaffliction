import torch
import argparse
from PIL import Image
from torch_classes import ImageDataset, CNN
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib as mpl
from Transformation import tranformations

mpl.rcParams['toolbar'] = 'None'
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

def load_image(image_pth: str):
    image = Image.open(image_pth).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    tensor = transform(image)
    return tensor

def main(src: str, image_pth: str, weights_pth: str, map_location="cpu"):
    print("Loading dataset")
    dataset = ImageDataset(src, (3, 128, 128))
    print("Dataset is loaded")

    model = CNN(NUM_OF_CLASSES, dataset.resize)
    model.load_state_dict(torch.load(weights_pth, map_location, weights_only=True))
    model.eval()

    img = load_image(image_pth)

    with torch.no_grad():
        input_tensor = img.unsqueeze(0)
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1)

    print(f"Prediction: {dataset.classes[prediction]}")

    try:
        jpg = Image.open(image_pth).convert('RGB')
        jpg2 = tranformations(image_pth)
    except Exception as e:
        print("\nError: can't transform base image")
        exit(-1)

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    fig.set_facecolor('black')
    axs[0].imshow(jpg)
    axs[0].axis('off')
    axs[0].set_facecolor('black')
    axs[0].set_anchor('N')

    axs[1].imshow(jpg2)
    axs[1].axis('off')
    axs[1].set_facecolor('black')
    axs[1].set_anchor('N')

    # Add the classification result as a separate text below the images
    fig.text(0.5, 0.2, "===      DL classification      ===", color='white',
             fontsize=14, ha='center', fontweight='bold')

    fig.text(0.40, 0.10, "Class predicted: ", color='white',
             fontsize=12, ha='center', fontweight='bold')

    fig.text(0.57, 0.10, dataset.classes[prediction], color='limegreen',
         fontsize=12, ha='center', fontweight='bold')

    plt.subplots_adjust(bottom=0.1)
    plt.show()


if __name__ == "__main__":
    try:
        args = args_parser()
        main(args.src, args.img, args.weight)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
