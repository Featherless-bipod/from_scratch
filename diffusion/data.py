import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os

class BridgeDataset(Dataset):
    def __init__(self, root, train=True, transform=None, source_class=None, target_class=None):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        
        if source_class is not None and target_class is not None:
            self.source_indices = [i for i, (_, label) in enumerate(self.dataset) if label == source_class]
            self.target_indices = [i for i, (_, label) in enumerate(self.dataset) if label == target_class]
            self.mode = 'bridge'
        else:
            self.mode = 'standard'

    def __len__(self):
        if self.mode == 'bridge':
            return min(len(self.source_indices), len(self.target_indices))
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.mode == 'bridge':
            img0, _ = self.dataset[self.source_indices[idx]]
            img1, _ = self.dataset[self.target_indices[idx]]
            return img0, img1
        
        #flow matching return 
        img, label = self.dataset[idx]
        return img, label

def get_dataloader(batch_size=128, image_size=32, source_class=None, target_class=None):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),#augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #scales[0, 1]to[-1, 1]
    ])

    dataset = BridgeDataset(root='./data', train=True, transform=transform, 
                            source_class=source_class, target_class=target_class)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True #for GPU transfer
    )
    
    return loader