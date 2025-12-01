"""
Z-Image-Turbo Gradio Web UI
An efficient 6B parameter image generation model with sub-second inference.
"""

import torch
import gradio as gr
from PIL import Image
import numpy as np
from diffusers import ZImagePipeline

# Global variable to store the pipeline
pipe = None

def load_model():
    """Load the Z-Image-Turbo model with optimizations"""
    global pipe
    if pipe is None:
        print("=" * 60)
        print("Loading Z-Image-Turbo model...")
        print("=" * 60)
        print("Model: Tongyi-MAI/Z-Image-Turbo (6B parameters)")
        print("Size: ~12GB (will download on first run)")
        print("This may take a few minutes on first launch...")
        print("=" * 60)
        
        # Use bfloat16 for optimal performance on supported GPUs
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        pipe.to("cuda")
        
        # Performance Optimizations
        print("Applying performance optimizations...")
        
        # 1. Try Flash Attention (if available)
        try:
            pipe.transformer.set_attention_backend("flash")
            print("✓ Flash Attention 2 enabled")
        except Exception as e:
            try:
                pipe.transformer.set_attention_backend("_flash_3")
                print("✓ Flash Attention 3 enabled")
            except Exception:
                print("✓ Using default SDPA attention (Flash Attention not available)")
        
        # 2. Enable torch.compile for faster inference (first run will be slower)
        try:
            print("Compiling model (first run will take longer)...")
            pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
            print("✓ Model compilation enabled")
        except Exception as e:
            print(f"✓ Model compilation skipped: {e}")
        
        # 3. Enable memory efficient attention
        try:
            pipe.enable_attention_slicing(1)
            print("✓ Attention slicing enabled")
        except Exception:
            pass
        
        # 4. Enable VAE slicing for memory efficiency
        try:
            pipe.enable_vae_slicing()
            print("✓ VAE slicing enabled")
        except Exception:
            pass
        
        # 5. Enable VAE tiling for large images
        try:
            pipe.enable_vae_tiling()
            print("✓ VAE tiling enabled")
        except Exception:
            pass
        
        print("=" * 60)
        print("Model loaded successfully with optimizations!")
        print("=" * 60)
        
    return pipe


def generate_image(
    prompt,
    width,
    height,
    num_inference_steps,
    guidance_scale,
    seed,
    use_random_seed
):
    """Generate image using Z-Image-Turbo"""
    try:
        # Load model if not already loaded
        print("Initializing generation...")
        pipeline = load_model()
        
        # Handle seed
        if use_random_seed:
            seed = np.random.randint(0, 2**32 - 1)
        
        generator = torch.Generator("cuda").manual_seed(int(seed))
        
        # Generate image
        print(f"Generating image...")
        print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Steps: {num_inference_steps}")
        print(f"  Seed: {seed}")
        
        image = pipeline(
            prompt=prompt,
            height=int(height),
            width=int(width),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        ).images[0]
        
        print("Image generated successfully!")
        return image, f"Generated successfully! Seed: {seed} | Resolution: {width}x{height} | Steps: {num_inference_steps}"
    
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        blank_image = Image.new('RGB', (512, 512), color='gray')
        return blank_image, error_msg


# Example prompts
example_prompts = [
    ["Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads.", 1024, 1024, 9, 0.0, 42, False],
    ["A serene mountain landscape at sunset, photorealistic, 8k, detailed", 1024, 1024, 9, 0.0, 123, False],
    ["A futuristic city with flying cars, cyberpunk style, neon lights, detailed architecture", 1024, 768, 9, 0.0, 456, False],
    ["A golden cat playing in a garden, sunlight, flowers, photorealistic", 768, 768, 9, 0.0, 789, False],
    ["Portrait of a wizard with a long white beard, magical staff, fantasy art, detailed", 1024, 1024, 9, 0.0, 101, False],
]


# Create Gradio interface
with gr.Blocks(title="Z-Image-Turbo Generator") as demo:
    gr.Markdown(
        """
        # Z-Image-Turbo Image Generator
        
        Generate high-quality images with **Z-Image-Turbo** - an efficient 6B parameter model with sub-second inference!
        
        **Features:**
        - Photorealistic Quality
        - Accurate Bilingual Text Rendering (Chinese & English)
        - Only 8 inference steps needed (set to 9)
        - Creative and detailed image generation
        
        **Important:** For Turbo models, keep guidance_scale at 0.0 for best results.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want to generate...",
                lines=5,
                value="Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern."
            )
            
            with gr.Row():
                width = gr.Slider(
                    minimum=512,
                    maximum=2048,
                    step=64,
                    value=1024,
                    label="Width"
                )
                height = gr.Slider(
                    minimum=512,
                    maximum=2048,
                    step=64,
                    value=1024,
                    label="Height"
                )
            
            with gr.Row():
                num_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=9,
                    label="Inference Steps (9 recommended)"
                )
                guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=0.0,
                    label="Guidance Scale (0.0 for Turbo)"
                )
            
            with gr.Row():
                seed = gr.Number(
                    label="Seed",
                    value=42,
                    precision=0
                )
                use_random_seed = gr.Checkbox(
                    label="Random Seed",
                    value=False
                )
            
            generate_btn = gr.Button("Generate Image", variant="primary", size="lg")
            
            gr.Markdown(
                """
                ### Tips:
                - Be specific and detailed in your prompts
                - Use 1024x1024 for best quality
                - Keep guidance_scale at 0.0 for Turbo models
                - 9 inference steps = 8 DiT forwards (optimal)
                """
            )
        
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Generated Image",
                type="pil",
                height=600
            )
            output_info = gr.Textbox(
                label="Generation Info",
                lines=2
            )
    
    gr.Markdown("## Example Prompts")
    gr.Examples(
        examples=example_prompts,
        inputs=[
            prompt,
            width,
            height,
            num_inference_steps,
            guidance_scale,
            seed,
            use_random_seed
        ],
        outputs=[output_image, output_info],
        fn=generate_image,
        cache_examples=False,
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
            use_random_seed
        ],
        outputs=[output_image, output_info]
    )
    
    gr.Markdown(
        """
        ---
        ### About Z-Image-Turbo
        
        Z-Image-Turbo is a powerful 6B parameter image generation model featuring:
        - Single-Stream Diffusion Transformer (S3-DiT) architecture
        - Decoupled-DMD acceleration algorithm
        - Sub-second inference on enterprise GPUs
        - Bilingual text rendering (Chinese & English)
        
        [ModelScope](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo) | [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
        """
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Z-Image-Turbo Gradio UI")
    print("=" * 60)
    print("Starting server...")
    print("Server will be available at: http://localhost:7860")
    print("Note: Model will download (~12GB) on first image generation")
    print("=" * 60)
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
