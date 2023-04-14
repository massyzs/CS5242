import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np
# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the dataset
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar10_dataset = load_dataset("cifar10")

trainset = cifar10_dataset["train"]
testset = cifar10_dataset["test"]

# breakpoint()
# Classes in the CIFAR-10 dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class CIFAR10Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.images = torch.tensor(np.array([np.array(img).reshape(3, 32, 32) for img in self.dataset["img"]]),dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.dataset["label"]),dtype=torch.long)
        # breakpoint()
        
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
            
        self.image=self.transform(self.images[idx])
       
        return self.image, self.labels[idx]
# Instantiate the model
net = SimpleCNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
train_set=CIFAR10Dataset(trainset,transform=transform)
    
test_set=CIFAR10Dataset(testset,transform=transform)
trainloader=DataLoader(train_set,batch_size=64,num_workers=8,shuffle=True,drop_last=True)
testloader=DataLoader(test_set,batch_size=64,num_workers=8,shuffle=False,drop_last=True)
# Train the model
num_epochs = 80
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    # Print average loss for the epoch
    print(f"Epoch: {epoch + 1}, Loss: {running_loss / (i + 1)}")

print("Training completed.")

# Evaluate the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")
