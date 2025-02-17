import os
import argparse
import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch_classes import ImageDataset, CNN, EarlyStopper
from tqdm import tqdm


torch.backends.cudnn.benchmark = True

# Hyperparams
CRITERION = nn.CrossEntropyLoss()
LR = 0.001
EPOCHS = 100
NUM_OF_CLASSES = 8
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
BATCH_SIZE = 128


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
    torch.manual_seed(42)
    generator = torch.Generator().manual_seed(42)

    dataset = ImageDataset(src, resize=(3, 128, 128))
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size

    cpu_count = os.cpu_count()
    num_workers = cpu_count - 1 if cpu_count > 1 else 0

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    img: torch.Tensor = dataset[0][0]

    model = CNN(NUM_OF_CLASSES, dataset.resize)
    print(f"Flattened size for fc1: {model._get_flattened_size()}")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        momentum=MOMENTUM
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=30,
    #     gamma=0.1
    # )
    total_step = len(train_dataloader)
    early_stopper = EarlyStopper()
    scaler = torch.amp.GradScaler()

    device = torch.device(0)
    model.to(device)

    for epoch in range(EPOCHS):
        loop = tqdm(
            train_dataloader,
            total=total_step,
            desc=f"Epoch [{epoch+1}/{EPOCHS}]"
        )
        model.train()
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(loop):
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = CRITERION(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Calculate training accuracy
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            accuracy = correct / total

            if "val_accuracy" and "val_loss" not in locals():
                val_accuracy = "Unk."
                val_loss = "Unk."

            loop.set_postfix(
                Loss=loss.item(),
                Acc=accuracy,
                Val_loss=val_loss,
                val_accuracy=val_accuracy,
            )

        # Validation
        # scheduler.step()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = CRITERION(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
        val_loss /= len(test_dataloader)
        val_accuracy = correct / total

        if early_stopper.early_stop(validation_loss=val_loss, model=model):
            print("Early stopped.")
            early_stopper.load_best_weights(model)
            break

    torch.save(model.state_dict(), "best_model.pth")
    print("Model weights saved to best_model.pth")

    config = {
        "input_size": dataset.resize,
        "num_classes": NUM_OF_CLASSES,
        "learning_rate": LR,
        "weight_decay": 0.005,
        "momentum": 0.9
    }

    with open("config.json", "w") as f:
        json.dump(config, f)
    print("Model configuration has been saved to config.json")


if __name__ == "__main__":
    args = args_parser()
    main(args.src)
