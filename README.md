# Vision Transformer Attention Visualization on CIFAR-10

This project trains a Vision Transformer (ViT) on the CIFAR-10 dataset and visualizes attention maps from the final layer.

## Files

- `train_vit.py`: Trains a ViT model on CIFAR-10
- `visualize_attention.py`: Visualizes attention maps for test images
- `notebook.ipynb`: Jupyter version of both scripts combined
- `requirements.txt`: Required Python packages
- `sample_output/`: Contains saved attention visualizations

## Sample Output

![Sample](sample_output/attention_image_1_label_3.png)

## How to Run

1. Install packages:
```bash
pip install -r requirements.txt
```
2.Train model:
```bash
python train_vit.py
```
3.Visualize attention:
```bash
python visualize_attention.py
```
