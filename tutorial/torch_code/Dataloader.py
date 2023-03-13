from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class Dataloader:
    def __init__(self, batch_size, random_seed, device):
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.device = device

    def get_data_loaders(self, dataset):
        # Get the paths of all image files
        train_imgs_paths = sorted(list(Path(dataset.train_img_dir).iterdir()))
        test_imgs_paths = sorted(list(Path(dataset.test_img_dir).iterdir()))

        idxs_train, idxs_val, idxs_test = dataset.get_random_partition_indices(self.random_seed)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # resize the images to 224x24 pixels
            transforms.ToTensor(),  # convert the images to a PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize the images color channels
        ])

        # The DataLoaders contain the ImageDataset class
        train_imgs_dataloader = DataLoader(
            ImageDataset(train_imgs_paths, idxs_train, transform, self.device),
            batch_size=self.batch_size
        )
        val_imgs_dataloader = DataLoader(
            ImageDataset(train_imgs_paths, idxs_val, transform, self.device),
            batch_size=self.batch_size
        )
        test_imgs_dataloader = DataLoader(
            ImageDataset(test_imgs_paths, idxs_test, transform, self.device),
            batch_size=self.batch_size
        )

        return train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader


class ImageDataset(Dataset):
    def __init__(self, imgs_paths, idxs, transform, device):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img).to(self.device)
        return img
