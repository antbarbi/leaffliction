import os
import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

class ImageDataset(Dataset):
    def __init__(self, root_dir: str, resize: tuple = (3, 64, 64)):
        self.root_dir = root_dir
        self.image_files = []
        self.classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self.resize = resize
        
        for root, dirs, files in os.walk(root_dir):
            if not dirs:
                for file in files:
                    class_name = os.path.basename(root)
                    self.image_files.append((file, class_name))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        name, label = self.image_files[index]
        img_path = os.path.join(self.root_dir, label, name)
        image = read_image(img_path, mode=ImageReadMode.RGB).float() / 255.0
        image = transforms.Resize(size=(64, 64))(image)
        label_idx = self.class_to_idx[label]
        one_hot_label = F.one_hot(torch.tensor(label_idx), num_classes=self.num_classes)
        return image, label_idx


class CNN(nn.Module):
    def __init__(self, num_of_classes: int, input_size: tuple):
        super(CNN, self).__init__()
        self.size = input_size

        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        def _get_flattened_size(self):
            with torch.no_grad():
                dummy_input = torch.zeros(1, *self.size)
                out = self.conv_layer1(dummy_input)
                out = self.pool1(out)
                out = self.conv_layer2(out)
                out = self.pool2(out)
                flattened_size = out.view(1, -1).size(1)
            return flattened_size

        self.fc1 = nn.Linear(_get_flattened_size(self), 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_of_classes)


    def _get_flattened_size(self, input_size=(3, 64, 64)):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            out = self.conv_layer1(dummy_input)
            out = self.pool1(out)
            out = self.conv_layer2(out)
            out = self.pool2(out)
            flattened_size = out.view(1, -1).size(1)
        return flattened_size


    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.pool1(out)

        out = self.conv_layer2(out)
        out = self.pool2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


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
    return parser.parse_args()


def main(src):
    # 1 - Get data ready (turned into tensors)
    dataset = ImageDataset(src)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
 
    cpu_count = os.cpu_count()
    num_workers = cpu_count - 1 if cpu_count > 1 else 0

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    img: torch.Tensor = dataset[0][0]

    model = CNN(8, dataset.resize)
    print(f"Flattened size for fc1: {model._get_flattened_size()}")
    # exit()

    # Hyperparams
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0005
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.005,
        momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.1
    )
    total_step = len(train_dataloader)
    patience = 10
    best_loss = float('inf')
    counter = 0
    epochs = 100

    device = torch.device(0)
    model.to(device)

    scaler = torch.amp.GradScaler()

    for epoch in range(epochs):
        loop = tqdm(train_dataloader, total=total_step, desc=f"Epoch [{epoch+1}/{epochs}]")
        model.train()
        for i, (images, labels) in enumerate(loop):
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if "val_loss" not in locals():
                val_loss = 0
            loop.set_postfix(
                    Loss=loss.item(),
                    Val_loss=val_loss,
                    Best_Loss=best_loss
            )
           # Validation

        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                with torch.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
        val_loss /= len(test_dataloader)

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    torch.save(model.state_dict(), "best_model.pth")
    print("Model weights saved to best_model.pth")

    config = {
        "input_size": dataset.resize,
        "num_classes": 8,
        "learning_rate": learning_rate,
        "weight_decay": 0.005,
        "momentum": 0.9
    }
    import json
    with open("config.json", "w") as f:
        json.dump(config, f)
    print("Model configuration has been saved to config.json")


if __name__ == "__main__":
    args = args_parser()
    main(args.src)
