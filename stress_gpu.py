import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

dataset = datasets.FakeData(
    size=1000,
    transform=transforms.ToTensor())
loader = DataLoader(
    dataset,
    num_workers=0,
    pin_memory=True
)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = models.resnet50()
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss().to(device)

for data, target in loader:
    data = data.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

print('Done')