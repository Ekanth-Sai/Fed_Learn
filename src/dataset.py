import os 
import pandas as pd 
from PIL import Image 
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
import sys 

LABEL_MAP = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6
}

NUM_CLASSES = 7

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225] 

def get_transforms(train = True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness = 0.2, contrast = 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD)
        ])
        
class HAM10000Dataset(Dataset):
    def __init__(self, csv_path: str, transform = None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform 
        
        assert "image_path" in self.df.columns, "CSV must have 'image_path' column"
        assert 'dx' in self.df.columns, "CSV must have 'dx' (diagnosis) column" 
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row =self.df.iloc[idx]
        image_path = row['image_path']
        label_str = row['dx']
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Could not open image as {image_path}: {e}")
        
        if self.transform:
            image = self.transform(image)
            
        label = LABEL_MAP[label_str]
        
        return image, label 
    
def get_client_dataloader(csv_path: str, batch_size: int = 16, train: bool = True):
    dataset = HAM10000Dataset(csv_path, transform = get_transforms(train = train)) 
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = train, num_workers = 2, pin_memory = False)
        
    return loader 
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage python src/dataset.py <path_to_client_csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    loader = get_client_dataloader(csv_path, batch_size = 6, train = True)
    images, labels = next(iter(loader))
    print(f"Batch image shape: {images.shape}")
    print(f"Batch labels: {labels}")
    print("Dataset loading works correctly")