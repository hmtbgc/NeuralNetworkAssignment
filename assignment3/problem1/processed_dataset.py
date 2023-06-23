from torchvision.datasets import STL10
from torchvision import transforms
from PIL import Image
import numpy as np

class STL10Pair(STL10):
    def __init__(self, root, download, train_num):
        super().__init__(root=root, split="unlabeled", download=download)
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])
        self.data = self.data[:train_num]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0))) 
        img1 = self.train_transform(img)
        img2 = self.train_transform(img)
        return img1, img2
        
