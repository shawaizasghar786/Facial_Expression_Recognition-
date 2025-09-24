import cv2, torch
from PIL import Image
from torchvision import transforms
from model import EmotionCNN

def run_webcam(class_names):
    model = EmotionCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load('model/emotion_model.pth'))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(), transforms.Resize((48, 48)), transforms.ToTensor()
    ])
    cap=cv2.VideoCapture(0)
    while True:
        ret, frame=cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tensor = transform(Image.fromarray(face)).unsqueeze(0)
        face=cv2.resize(gray,(48,48))
        output=model(tensor)
        pred = torch.argmax(output, dim=1).item()
        cv2.putText(frame, class_names[pred], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
        