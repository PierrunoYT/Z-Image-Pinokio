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
    // Install Flash Attention for better performance
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "pip install flash-attn --no-build-isolation"
        ],
      }
    },
    {
      method: "notify",
      params: {
        html: "Installation complete! Click 'Start' to launch Z-Image-Turbo with xformers and Flash Attention."
      }
    }
  ]
}
