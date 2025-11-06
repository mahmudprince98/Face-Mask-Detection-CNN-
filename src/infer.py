import argparse, yaml, torch
from PIL import Image
from torchvision import transforms
from model import create_model

def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    classes = ckpt["classes"]
    return model, classes

def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def predict(image_path, model, device, img_size):
    tfm = build_transform(img_size)
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = probs.argmax().item()
    return pred, probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--weights", type=str, default="models/best_mask_cnn.pt")
    args = parser.parse_args()

    with open("src/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=2).to(device)
    model, classes = load_checkpoint(model, args.weights, device)

    pred, probs = predict(args.image, model, device, cfg["data"]["img_size"])
    label = classes[pred]
    print(f"Prediction: {label} | Probabilities: {dict(zip(classes, [float(f'{p:.4f}') for p in probs]))}")
