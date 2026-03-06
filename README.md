# 📷 Vision Transformer (ViT) From Scratch

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Transformers](https://img.shields.io/badge/Transformer-Vision-important)
![License](https://img.shields.io/badge/License-MIT-green)

A **minimal implementation of a Vision Transformer (ViT)** using **PyTorch**, designed for learning and experimentation.

This project demonstrates how transformers — originally created for NLP — can be applied to **image classification tasks**.

Inspired by the research paper  
**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**

---

# 🧠 What is a Vision Transformer?

A **Vision Transformer (ViT)** replaces convolutional neural networks with a **transformer architecture**.

Instead of processing pixels with convolution filters:

1️⃣ The image is split into **patches**  
2️⃣ Patches are **flattened and embedded**  
3️⃣ A **Transformer Encoder** processes them  
4️⃣ The **[CLS] token** predicts the image class

---

# 🏗 Architecture Overview

```
Input Image
     │
     ▼
Split Into Patches
     │
     ▼
Patch Embedding Layer
     │
     ▼
Add Positional Encoding
     │
     ▼
Transformer Encoder Blocks
     │
     ▼
Classification Head
     │
     ▼
Prediction
```

---

# 📂 Project Structure

```
Vision-Transformer/
│
├── ViT.ipynb           # Main notebook implementation
├── README.md
└── images/
    └── vit_architecture.png
```

---

# ⚡ Quick Start

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/solo938/Vision-Transformer.git

cd Vision-Transformer
```

---

## 2️⃣ Install Dependencies

```bash
pip install torch torchvision matplotlib numpy
```

---

## 3️⃣ Run the Notebook

```bash
jupyter notebook ViT.ipynb
```

or open with **Google Colab**

---

# 🧩 Core Components Implemented

## Patch Embedding

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, emb_dim):
        super().__init__()

        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
```

---

## Multi-Head Self Attention

```python
class MultiHeadAttention(nn.Module):

    def __init__(self, emb_dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.scale = emb_dim ** -0.5

        self.qkv = nn.Linear(emb_dim, emb_dim * 3)
        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):

        B, N, C = x.shape
        qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2,0,3,1,4)

        q,k,v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.transpose(1,2).reshape(B,N,C)

        return self.proj(out)
```

---

# 📊 Vision Transformer Pipeline

```
Image (224x224)
      │
      ▼
Patchify (16x16 patches)
      │
      ▼
196 tokens
      │
      ▼
Linear Embedding
      │
      ▼
Add Positional Encoding
      │
      ▼
Transformer Encoder × L
      │
      ▼
MLP Head
      │
      ▼
Class Prediction
```

---

# 📈 Possible Improvements

- Train on **CIFAR-10**
- Add **Data Augmentation**
- Add **Attention Visualization**
- Convert notebook → **Modular PyTorch project**
- Add **training scripts**

---

# 🧪 Example Datasets

You can test this implementation on:

- CIFAR-10  
- ImageNet  
- MNIST  

---

# 📚 Learning Resources

If you want to deeply understand Vision Transformers:

- **Attention Is All You Need**
- **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**

Libraries to explore:

- PyTorch  
- Hugging Face Transformers  
- timm (PyTorch Image Models)

---

# 🛣 Roadmap

- [ ] Implement training loop  
- [ ] Add dataset loader  
- [ ] Implement ViT-B/16  
- [ ] Add visualization of attention maps  
- [ ] Convert notebook → production code

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository  
2. Create a new branch  
3. Submit a pull request

---

# ⭐ Support

If you found this useful:

⭐ Star the repository  
🍴 Fork it  
🧠 Experiment with your own transformer ideas

---

# 📜 License

MIT License
