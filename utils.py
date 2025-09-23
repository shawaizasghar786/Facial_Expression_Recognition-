from torchvision import datasets,transforms
from torch.utils.data import Dataloader

def get_dataloaders(data_dir,batch_size=64):
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48,48)),
        transforms.ToTsnsor()
    ])
    train_data = datasets.ImageFolder(root=f"{base_path}/train", transform=transform)
    test_data = datasets.ImageFolder(root=f"{base_path}/test", transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_data.classes