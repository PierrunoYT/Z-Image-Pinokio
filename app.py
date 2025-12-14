import torch
import gradio as gr
from diffusers import ZImagePipeline
import random
from datetime import datetime
import uuid
import os

# Global pipeline variable
pipe = None
output_dir = "outputs"

# Create outputs directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_pipeline():
    global pipe
    if pipe is None:
        print("Loading Z-Image-Turbo pipeline...")
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        pipe.to("cuda")
        print("Pipeline loaded successfully!")
    return pipe

def generate_image(
    prompt,
    width,
    height,
    num_inference_steps,
    guidance_scale,
    seed,
    randomize_seed,
):
    pipe = load_pipeline()
    
    if randomize_seed:
        seed = random.randint(0, 2**32 - 1)
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    
    # Save with unique timestamp + UUID filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    filename = os.path.join(output_dir, f"image_{timestamp}_{unique_id}.png")
    image.save(filename)
    
    return filename, seed

# Pre-load the pipeline on startup
load_pipeline()

# Create Gradio interface
with gr.Blocks(title="Z-Image-Turbo") as demo:
    gr.Markdown(
        """
        # 🎨 Z-Image-Turbo
        Generate high-quality images using the Z-Image-Turbo model from Tongyi-MAI.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want to generate...",
                lines=3,
            )
            
            # Resolution presets
            gr.Markdown("**Resolution Presets:**")
            with gr.Row():
                preset_512 = gr.Button("512×512", size="sm")
                preset_768 = gr.Button("768×768", size="sm")
                preset_1024 = gr.Button("1024×1024", size="sm")
                preset_landscape = gr.Button("1024×768", size="sm")
                preset_portrait = gr.Button("768×1024", size="sm")
            
            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=512,
                    maximum=2048,
                    step=64,
                    value=1024,
                )
                height = gr.Slider(
                    label="Height",
                    minimum=512,
                    maximum=2048,
                    step=64,
                    value=1024,
                )
            
            with gr.Row():
                num_inference_steps = gr.Slider(
                    label="Inference Steps",
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=9,
                    info="9 steps is recommended for Turbo model",
                )
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=0.0,
                    info="0.0 is recommended for Turbo model",
                )
            
            with gr.Row():
                seed = gr.Number(
                    label="Seed",
                    value=42,
                    precision=0,
                )
                randomize_seed = gr.Checkbox(
                    label="Randomize Seed",
                    value=True,
                )
            
            generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Generated Image",
                type="filepath",
                format="png",
            )
            used_seed = gr.Number(label="Seed Used", interactive=False)
    
    gr.Examples(
        examples=[
            ["Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Soft-lit outdoor night background, silhouetted tiered pagoda, blurred colorful distant lights."],
            ["A majestic dragon soaring through clouds at sunset, scales shimmering with gold and crimson light, photorealistic, 8k, highly detailed"],
            ["A cozy coffee shop interior, warm lighting, plants on shelves, exposed brick walls, steaming latte on wooden table, rainy window view"],
            ["Cyberpunk city street at night, neon signs in Japanese, rain-slicked pavement reflecting lights, flying cars overhead, cinematic"],
            ["Portrait of an astronaut in a detailed spacesuit, Earth visible through helmet reflection, dramatic lighting, hyperrealistic"],
        ],
        inputs=[prompt],
        label="Example Prompts",
    )
    
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt,
            width,
            height,
            num_inference_steps,
            guidance_scale,
            seed,
            randomize_seed,
        ],
        outputs=[output_image, used_seed],
    )
    
    # Preset buttons event handlers
    preset_512.click(fn=lambda: (512, 512), outputs=[width, height])
    preset_768.click(fn=lambda: (768, 768), outputs=[width, height])
    preset_1024.click(fn=lambda: (1024, 1024), outputs=[width, height])
    preset_landscape.click(fn=lambda: (1024, 768), outputs=[width, height])
    preset_portrait.click(fn=lambda: (768, 1024), outputs=[width, height])

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=False,
        show_error=True
    )
