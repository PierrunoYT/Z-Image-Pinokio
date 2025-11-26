module.exports = {
  daemon: true,
  run: [
    // Launch Z-Image-Turbo Gradio Web UI
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: { },
        message: [
          "python app.py"
        ],
        on: [{
          // Monitor for Gradio's HTTP URL output
          "event": "/http:\/\/\\S+/",
          "done": true
        }]
      }
    },
    // Set the local URL variable for the "Open Web UI" button
    {
      method: "local.set",
      params: {
        url: "{{input.event[0]}}"
      }
    },
    {
      method: "notify",
      params: {
        html: "Z-Image-Turbo is running! Click 'Open Web UI' to start generating images."
      }
    }
  ]
}

