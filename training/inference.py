import torch
from torchvision import transforms
from PIL import Image
from model.crnn import CRNN
from utils import LabelConverter

charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
converter = LabelConverter(charset)

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

device = torch.device('cpu')
model = CRNN(32, 1, len(charset) + 1, 256).to(device)
model.load_state_dict(torch.load('crnn_iam.pth', map_location=device))
model.eval()

image = Image.open('science_fair_2025-main/handwritten_text.png').convert('L')
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    decoded = converter.decode(output)

with open('output_text.txt', 'w') as f:
    f.write(decoded[0])
