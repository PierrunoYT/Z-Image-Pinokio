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
    {
      method: "notify",
      params: {
        html: "Installation complete! Click 'Start' to launch Z-Image-Turbo."
      }
    }
  ]
}
