# Z-Image-Turbo Pinokio

⚡️ **Z-Image-Turbo** - Efficient 6B parameter image generation model with sub-second inference, packaged for [Pinokio](https://pinokio.computer/).

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

## 🌟 Features

- **⚡️ Ultra-Fast Generation**: Only 8 inference steps needed (sub-second on enterprise GPUs)
- **📸 Photorealistic Quality**: Strong photorealistic image generation with excellent aesthetic quality
- **📖 Bilingual Text Rendering**: Excels at rendering complex Chinese and English text
- **🎨 Advanced Architecture**: Single-Stream Diffusion Transformer (S3-DiT) with Decoupled-DMD
- **🚀 Optimized Performance**: Includes xformers and Flash Attention support
- **🖥️ Easy to Use**: Beautiful Gradio web UI with one-click installation via Pinokio

## 📦 What's Included

- **Gradio Web UI** (`app.py`) - User-friendly interface for image generation
- **Pinokio Integration** - One-click installation and launch
- **Performance Optimizations** - xformers and Flash Attention enabled by default
- **Example Prompts** - Pre-loaded examples to get you started

## 🚀 Quick Start

### Via Pinokio (Recommended)

1. Install [Pinokio](https://pinokio.computer/)
2. Search for "Z-Image-Turbo" in Pinokio
3. Click **Install**
4. Click **Start**
5. Click **Open Web UI**

### Manual Installation

```bash
# Clone the repository
git clone <repository-url>
cd Z-Image-Pinokio

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA
pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128

# Install Flash Attention
pip install flash-attn --no-build-isolation

# Run the app
python app.py
```

Then open `http://localhost:7860` in your browser.

## 🎯 Usage

1. **Enter a prompt** describing the image you want to generate
2. **Adjust settings** (optional):
   - Resolution: 512-2048px (default: 1024x1024)
   - Inference Steps: 1-20 (default: 9 = 8 DiT forwards)
   - Guidance Scale: Keep at 0.0 for Turbo models
   - Seed: For reproducible results
3. **Click "Generate Image"**
4. **Wait a few seconds** for your image!

## 📝 Tips for Best Results

- **Be Specific**: Detailed prompts produce better results
- **Optimal Settings**: Use 1024x1024 resolution, 9 inference steps, guidance_scale 0.0
- **Style Keywords**: Include art style, lighting, mood descriptors
- **Text Rendering**: Model excels at Chinese and English text in images
- **Guidance Scale**: Must be 0.0 for Turbo models

## 🏗️ Model Architecture

**Z-Image-Turbo** uses:
- **S3-DiT**: Scalable Single-Stream Diffusion Transformer
- **Decoupled-DMD**: Core acceleration algorithm enabling 8-step generation
- **DMDR**: Fusing DMD with Reinforcement Learning
- **6B Parameters**: Optimal balance of quality and efficiency

## 📊 Performance

- Sub-second inference latency on enterprise GPUs
- State-of-the-art results on AI Arena leaderboard
- Photorealistic quality with bilingual text rendering
- Only 8 NFE (Number of Function Evaluations)

## 🔧 System Requirements

### Minimum
- **GPU**: NVIDIA RTX 3060 (12GB VRAM)
- **RAM**: 16GB
- **Storage**: 20GB free space
- **OS**: Windows 10/11, Linux, macOS

### Recommended
- **GPU**: NVIDIA RTX 4090 or A100 (24GB+ VRAM)
- **RAM**: 32GB
- **Storage**: 50GB free space

## 📚 Model Information

- **Model**: [Tongyi-MAI/Z-Image-Turbo](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo)
- **Parameters**: 6B
- **Architecture**: Single-Stream Diffusion Transformer
- **Inference Steps**: 8 (set num_inference_steps to 9)
- **Guidance**: 0.0 (required for Turbo models)

## 🔗 Links

- 🌐 [ModelScope](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo)
- 🤗 [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- 💻 [Pinokio](https://pinokio.computer/)

## 📄 License

This project is licensed under the Apache License 2.0. See the model page for more details.

## 🙏 Credits

- **Model**: Tongyi-MAI Team
- **Framework**: ModelScope, Gradio
- **Deployment**: Pinokio

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check the [ModelScope page](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo)
- Visit the Pinokio community

---

**Made with ❤️ for the AI community**

