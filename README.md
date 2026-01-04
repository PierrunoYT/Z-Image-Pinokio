# Z-Image-Turbo for Pinokio

A Gradio web interface for the Z-Image-Turbo model, designed for easy installation via [Pinokio](https://pinokio.computer/).

## Features

- **One-Click Install** - Pinokio handles all dependencies automatically
- **Modern Gradio UI** - Clean interface with all generation options
- **High-Quality Images** - Uses the Z-Image-Turbo model from Tongyi-MAI
- **Fast Generation** - Turbo model generates images in just 8 steps
- **Full Control** - Adjustable dimensions, steps, guidance, and seed

## Installation

### Via Pinokio (Recommended)

1. Install [Pinokio](https://pinokio.computer/)
2. Search for "Z-Image-Turbo" or add this repository URL
3. Click **Install** - Pinokio will:
   - Create a Python virtual environment
   - Install PyTorch with CUDA support
   - Install diffusers from source (required for ZImagePipeline)
   - Download the Z-Image-Turbo model (~12GB)
4. Click **Start** to launch the web UI

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/PierrunoYT/Z-Image-Pinokio.git
cd Z-Image-Pinokio

# Create virtual environment
python -m venv env
source env/bin/activate  # Linux/Mac
# or: env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/huggingface/diffusers

# Run the app
python app.py
```

## Usage

1. **Start** the app via Pinokio or `python app.py`
2. Open `http://localhost:7860` in your browser
3. Enter a prompt describing your image
4. Adjust settings:
   - **Width/Height**: 512-2048px (default 1024x1024)
   - **Inference Steps**: 1-20 (default 9, recommended for Turbo)
   - **Guidance Scale**: 0.0-10.0 (default 0.0 for Turbo model)
   - **Seed**: Set specific seed or randomize
5. Click **Generate**

## Settings Guide

| Setting | Default | Description |
|---------|---------|-------------|
| Width/Height | 1024 | Image dimensions (512-2048px) |
| Inference Steps | 9 | Number of denoising steps (8 actual DiT forwards) |
| Guidance Scale | 0.0 | CFG scale (0.0 recommended for Turbo) |
| Seed | Random | Reproducibility control |

## System Requirements

### Minimum
- **GPU**: 12GB VRAM (NVIDIA RTX 30-series or newer with CUDA)
- **RAM**: 16GB
- **Storage**: ~15GB for model
- **Driver**: NVIDIA Driver 535.x or newer (for CUDA 12.4 support)

### Recommended
- **GPU**: 16GB+ VRAM (RTX 3090, 4090, etc.)
- **RAM**: 32GB
- **Storage**: SSD for faster loading

### Supported GPUs
- ✅ RTX 40-series (4090, 4080, 4070, etc.)
- ✅ RTX 30-series (3090, 3080, 3070, 3060 12GB)
- ✅ RTX 20-series (2080 Ti, 2080, 2070)
- ✅ Tesla/Quadro series with 12GB+ VRAM

## Troubleshooting

### CUDA Errors
If you encounter "CUDA error: no kernel image is available for execution on the device":
1. **Update NVIDIA Driver**: Install the latest driver from [NVIDIA's website](https://www.nvidia.com/drivers)
2. **Reinstall**: Click "Reset" then "Install" in Pinokio to reinstall with compatible PyTorch version
3. **Verify GPU**: Run `nvidia-smi` to confirm your GPU is detected

### Out of Memory Errors
If you get CUDA out of memory errors:
1. Close other GPU-intensive applications
2. Reduce image resolution (try 768x768 or 512x512)
3. Restart the application

## Model Info

- **Model**: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- **License**: Apache 2.0
- **Size**: ~12GB

## Credits

- **Z-Image-Turbo**: [Tongyi-MAI (Alibaba)](https://huggingface.co/Tongyi-MAI)
- **Diffusers**: [Hugging Face](https://github.com/huggingface/diffusers)
- **Pinokio**: [Pinokio](https://pinokio.computer/)

## License

MIT
