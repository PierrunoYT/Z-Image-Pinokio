module.exports = {
  run: [
    // Install Z-Image-Turbo dependencies from requirements.txt
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "pip install -r requirements.txt"
        ],
      }
    },
    // Install diffusers from source (required for ZImagePipeline)
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "pip install git+https://github.com/huggingface/diffusers"
        ],
      }
    },
    // Install PyTorch with CUDA support and xformers
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          xformers: true   // Enable xformers for memory-efficient attention
        }
      }
    },
    // Install build dependencies for Flash Attention
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "pip install wheel packaging ninja"
        ],
      }
    },
    // Install Flash Attention (optional - will skip if build fails)
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "pip install flash-attn --no-build-isolation || echo Flash Attention installation failed, continuing without it..."
        ],
      }
    },
    {
      method: "notify",
      params: {
        html: "Installation complete! Click 'Start' to launch Z-Image-Turbo.<br>Note: Flash Attention is optional and may not install on all systems."
      }
    }
  ]
}
