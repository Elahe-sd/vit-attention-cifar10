# ✅ Step 1: Imports
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from transformers import ViTFeatureExtractor, ViTForImageClassification

# ✅ Step 2: Load feature extractor for ViT
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# ✅ Step 3: Define transform for CIFAR-10 to match ViT input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# ✅ Step 4: Load CIFAR-10 dataset
train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16)

# ✅ Step 5: Define label mappings manually
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']
id2label = {str(i): label for i, label in enumerate(labels)}
label2id = {label: str(i) for i, label in enumerate(labels)}

# ✅ Step 6: Load ViT model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=10,
    id2label=id2label,
    label2id=label2id
)
model.config.output_attentions = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Step 7: Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# ✅ Step 8: Training loop
for epoch in range(2):
    model.train()
    total_loss = 0
    correct = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=images)
        loss = criterion(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()

    accuracy = correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {accuracy:.4f}")
    torch.save(model.state_dict(), "vit_cifar10_weights.pth")


