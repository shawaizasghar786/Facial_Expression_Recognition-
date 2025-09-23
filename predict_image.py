from PIL import Image
import torch
from torchvision import transforms
from model import EmotionCNN

def predict_image(path, class_names):
    model = EmotionCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load('model/emotion_model.pth'))
    model.eval()

    transform=transforms.Compose([
        transforms.Grayscale(),transforms.Resize((48,48)),transforms.ToTensor()
    ])
    image=Image.open(path).convert('L')
    input_tensor=transform(image).unsqueeze(0)
    output=model(input_tensor)
    pred=torch.argmax(output, dim=1).item()
    print(f"Prediction: {class_names[pred]}")