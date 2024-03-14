from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
class Predict_dataset(Dataset):
    def __init__(self, img_path, transform = None):
        self.img_path  = list(Path(img_path).glob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        img = self.transform(img)
        img = img.unsqueeze(0)
        return img