import torch
import gradio as gr
from diffusers import ZImagePipeline
import random

# Global pipeline variable
pipe = None

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
    negative_prompt,
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
        negative_prompt=negative_prompt if negative_prompt else None,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    
    return image, seed

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
            negative_prompt = gr.Textbox(
                label="Negative Prompt (Optional)",
                placeholder="What to avoid in the image...",
                lines=2,
            )
            
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
                type="pil",
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
            negative_prompt,
            width,
            height,
            num_inference_steps,
            guidance_scale,
            seed,
            randomize_seed,
        ],
        outputs=[output_image, used_seed],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
