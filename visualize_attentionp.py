import random
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
import matplotlib.pyplot as plt
import os

def visualize_vit_attention(model, image_tensor, feature_extractor, device, layer=-1, head=0, save_path=None):
    model.eval()
    with torch.no_grad():
        # Prepare input batch (add batch dimension)
        image = image_tensor.unsqueeze(0).to(device)

        # Forward pass with output attentions
        outputs = model(pixel_values=image, output_attentions=True)

        attentions = outputs.attentions  # List of attention maps: (batch, heads, tokens, tokens)
        attn = attentions[layer][0, head, 0, 1:]  # CLS token's attention to patches
        num_patches = int(attn.shape[0] ** 0.5)
        attn = attn.reshape(num_patches, num_patches).cpu()

        # Original image (unnormalize for visualization)
        img = image_tensor.permute(1, 2, 0).cpu()
        img = img * torch.tensor(feature_extractor.image_std) + torch.tensor(feature_extractor.image_mean)
        img = img.clamp(0, 1).numpy()

        plt.figure(figsize=(10,5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.imshow(attn, cmap='inferno', alpha=0.6)
        plt.title(f'Attention Layer {layer} Head {head}')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()

# Setup
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=10
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)

# Directory to save images
save_dir = "./attention_outputs"
os.makedirs(save_dir, exist_ok=True)

num_images_to_visualize = 5  # Change this to however many you want

for i in range(num_images_to_visualize):
    idx = random.randint(0, len(test_dataset) - 1)
    image_tensor, label = test_dataset[idx]
    print(f"Visualizing image idx {idx} with label {label}")

    save_path = os.path.join(save_dir, f"attention_image_{idx}_label_{label}.png")
    visualize_vit_attention(model, image_tensor, feature_extractor, device, layer=-1, head=0, save_path=save_path)
    
    
    
    
    
    


# Pick a random image from test set
idx = random.randint(0, len(test_dataset) - 1)
image_tensor, label = test_dataset[idx]

print(f"Selected image index: {idx}, Label: {label}")

# Visualize attention for the selected image (using the function from previous message)
visualize_vit_attention(model, image_tensor, feature_extractor, device, layer=-1, head=0, save_path='attention.png')


