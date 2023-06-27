from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image

class CIFAR10Pair(CIFAR10):
    def __init__(self, root, download, train):
        super().__init__(root=root, download=download, train=train)
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        img1 = self.train_transform(img)
        img2 = self.train_transform(img)
        return img1, img2
    
        