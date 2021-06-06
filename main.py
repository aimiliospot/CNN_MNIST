import torch
import torch.utils.data
import torch.nn as nn
from torchvision import datasets, transforms
import CnnClasses
from tqdm.notebook import tqdm, trange


#Mnist data load
train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)

#Create batchesd for training
trainBatches = torch.utils.data.DataLoader(train, batch_size= 100, shuffle= True)
testBatches = torch.utils.data.DataLoader(test, batch_size= 100, shuffle= False)

#Training
model = CnnClasses.CnnMnist()
lossCr = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in trange(3):
    for images, labels in tqdm(trainBatches):
        optimizer.zero_grad()

        x = images
        y = model(x)
        loss = lossCr(y, labels)
        loss.backward()
        optimizer.step()

## Testing
correct = 0
total = len(test)

with torch.no_grad():
    # Iterate through test set minibatchs
    for images, labels in tqdm(testBatches):
        # Forward pass
        x = images  # <---- change here
        y = model(x)

        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

print('Test accuracy: {}'.format(correct/total))
