import os
import glob
import numpy as np
from scipy import misc

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

        
class ToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)

class dataset(Dataset):
    def __init__(self, root_dir, dataset_type, img_size, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_names = [f for f in glob.glob(os.path.join(root_dir, "*", "*.npz")) \
                            if dataset_type in f]
        self.img_size = img_size
        # self.embeddings = np.load('./embedding.npy')

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # data_path = os.path.join(self.root_dir, self.file_names[idx])
        data_path = self.file_names[idx]
        data = np.load(data_path)
        image = data["image"].reshape(16, 160, 160)
        resize_image = []
        for idx in range(0, 16):
            resize_image.append(misc.imresize(image[idx,:,:], (self.img_size, self.img_size)))
        resize_image = np.stack(resize_image)
        # image = resize(image, (16, 128, 128))
        # meta_matrix = data["mata_matrix"]
        target = data["target"]
        # structure = data["structure"]
        meta_target = data["meta_target"]

        if meta_target.dtype == np.int8:
            meta_target = meta_target.astype(np.uint8)
        # if meta_structure.dtype == np.int8:
        #     meta_structure = meta_structure.astype(np.uint8)
    
        del data
        if self.transform:
            resize_image = self.transform(resize_image)
            # meta_matrix = self.transform(meta_matrix)
            target = torch.tensor(target, dtype=torch.long)
            meta_target = self.transform(meta_target)
            # meta_structure = self.transform(meta_structure)
            # meta_target = torch.tensor(meta_target, dtype=torch.long)
        return resize_image, target, meta_target
