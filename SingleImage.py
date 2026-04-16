import torch
from PIL import Image
from torchvision import transforms

from model import build_model


def load_trained_model(model_path):
    model, device = build_model()
    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 224)

    return model, class_names, image_size, device


def predict_image(image_path, model_path="models/densenet_skin_cancer.pth"):
    model, class_names, image_size, device = load_trained_model(model_path)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    result = {
        "predicted_class": class_names[pred_idx],
        "confidence": float(probs[pred_idx].item()),
        "probabilities": {
            class_names[i]: float(probs[i].item()) for i in range(len(class_names))
        }
    }
    return result


if __name__ == "__main__":
    result = predict_image("test.jpg")
    print(result)