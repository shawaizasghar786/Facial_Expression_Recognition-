import torch
import torch.nn as nn
import torch.optim as optim
from model import EmotionCNN
from utils import get_dataloaders


train_loader, test_loader, class_names = get_dataloaders()

model = EmotionCNN(num_classes=len(class_names))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(15):
    for image, labels in train_loader:
        output=model(image)
        loss=criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item(): .4f}")
torch.save(model.state_dist(),'model/emotion_model.pth')