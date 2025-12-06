module.exports = {
  run: [
    // Install PyTorch with CUDA support first
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
    // Install Z-Image-Turbo dependencies from requirements.txt
    // This includes diffusers from source with ZImagePipeline support
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "pip install -r requirements.txt"
        ],
      }
    },
    // Pre-download Z-Image-Turbo model (~12GB)
    // Using && dir to ensure clean prompt termination
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "huggingface-cli download Tongyi-MAI/Z-Image-Turbo --local-dir-use-symlinks False && dir"
        ],
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
