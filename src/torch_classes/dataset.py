import os
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, root_dir: str, resize: tuple = (3, 64, 64)):
        self.root_dir = root_dir
        self.image_files = []
        self.classes = sorted(
            entry.name for entry in os.scandir(root_dir)
            if entry.is_dir()
        )
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }
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
        return image, label_idx
