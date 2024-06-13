import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import datetime
import numpy as np
from tqdm import tqdm
import torchvision.models as models


bit_lengths = [8]
model_path = 'Automated_Output'

fig_path = 'figures'

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=2)

classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

# device = torch.device('cuda')
device = 'cpu'


def create_model(model_path):
    vgg16 = models.vgg16(pretrained=False)
    vgg16.classifier[4] = nn.Linear(4096,1024)
    vgg16.classifier[6] = nn.Linear(1024,10)
    vgg16.load_state_dict(torch.load(f'{model_path}',map_location=device))
    vgg16.eval()
    return vgg16

print(device)
vgg16 = create_model(model_path)
vgg16.to(device)
correct = 0
total = 0
with torch.no_grad():
    for data in tqdm(testloader):
        images, labels = data[0].to(device), data[1].to(device)
        outputs = vgg16(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

