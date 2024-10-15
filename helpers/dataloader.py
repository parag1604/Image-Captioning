import os
import torch
import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class Flickr8kDataset(Dataset):
    def __init__(
            self, images_dir, captions_path, transform):
        self.images_dir = images_dir
        self.image_paths = []
        self.captions_path = captions_path
        self.captions = self.__load_captions()
        self.images = [item for item in self.captions.keys()]
        self.transform = transform

    def __load_captions(self):
        return pickle.load(open(self.captions_path, 'rb'))

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        captions = self.captions[image_path]
        image_path = os.path.join(self.images_dir, image_path)
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        return image, captions


def get_dataset(transform, data_path, caption_path):
    dataset = Flickr8kDataset(data_path, caption_path, transform)
    return dataset


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers)
