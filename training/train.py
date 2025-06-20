import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model.crnn import CRNN
from utils import LabelConverter
import os

class OCRDataset(Dataset):
    def __init__(self, label_file, transform, converter):
        with open(label_file, 'r') as f:
            lines = f.read().splitlines()
        self.samples = []
        self.transform = transform
        self.converter = converter
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(label_file)))
        

        for line in lines:
            try:
                img_path, label = line.split(maxsplit=1)
                if img_path.startswith('data/'):
                    img_path = img_path[5:]
                
                
                parts = img_path.split(os.sep)
                if len(parts) >= 3:

                    img_filename = parts[3]
                    filename_parts = img_filename.split('-')
                    if len(filename_parts) >= 3:
                        # prefix/numb
                        prefix = filename_parts[0]
                        number = filename_parts[1]
                        # letter exists
                        letter = ''
                        if len(filename_parts[2]) > 2:
                            letter = filename_parts[2][0]
                        # directory
                        correct_dir = f"{prefix}-{number}{letter}"
                        # update
                        parts[2] = correct_dir
                
                full_img_path = os.path.join(self.base_dir, 'data', *parts)
                if not os.path.exists(full_img_path):
                    print(f"Warning: File does not exist: {full_img_path}")
                    continue      
                if os.path.getsize(full_img_path) == 0:
                    print(f"Warning: File is empty: {full_img_path}")
                    continue
                

                try:
                    with Image.open(full_img_path) as img:
                        img.verify()  
                except Exception as e:
                    print(f"Warning: Invalid image file {full_img_path}: {str(e)}")
                    continue
                
                self.samples.append((img_path, label))
            except Exception as e:
                print(f"Warning: Error processing line: {line}")
                print(f"Error: {str(e)}")
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        if img_path.startswith('data/'):
            img_path = img_path[5:]
        
        
        parts = img_path.split(os.sep)
        if len(parts) >= 3:
            
            img_filename = parts[3]
           
            filename_parts = img_filename.split('-')
            if len(filename_parts) >= 3:
       
                prefix = filename_parts[0]
                number = filename_parts[1]
                letter = ''
                if len(filename_parts[2]) > 2:
                    letter = filename_parts[2][0]
                correct_dir = f"{prefix}-{number}{letter}"
                parts[2] = correct_dir
        
        full_img_path = os.path.join(self.base_dir, 'data', *parts)
        
        try:
            if not os.path.exists(full_img_path):
                raise FileNotFoundError(f"File does not exist: {full_img_path}")
                
            if os.path.getsize(full_img_path) == 0:
                raise ValueError(f"File is empty: {full_img_path}")
            
            image = Image.open(full_img_path).convert('L')
            image = self.transform(image)
            target = torch.tensor(self.converter.encode(label), dtype=torch.long)
            return image, target, len(target)
        except Exception as e:
            print(f"Error loading image {full_img_path}: {str(e)}")
            empty_image = torch.zeros((1, 32, 128))
            empty_target = torch.tensor([0], dtype=torch.long)
            return empty_image, empty_target, 1

def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)
    labels = torch.cat(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, lengths

charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
converter = LabelConverter(charset)

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = OCRDataset('training/data/labels.txt', transform, converter)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN(32, 1, len(charset) + 1, 256).to(device)
criterion = nn.CTCLoss(blank=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)
epoch_amount = 10



for epoch in range(epoch_amount):
    model.train()
    epoch_loss = 0
    for images, labels, lengths in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        input_lengths = torch.full((images.size(0),), images.size(3) // 4, dtype=torch.long)
        input_lengths = torch.clamp(input_lengths, max=31)

        outputs = model(images).log_softmax(2)
        loss = criterion(outputs, labels, input_lengths, lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), 'crnn_iam.pth')
