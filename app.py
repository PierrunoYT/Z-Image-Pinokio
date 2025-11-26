import torch
import gradio as gr
from PIL import Image
import numpy as np
from diffusers import DiffusionPipeline
import sys
import io

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Global variable to store the pipeline
pipe = None

def load_model():
    """Load the Z-Image-Turbo model"""
    global pipe
    if pipe is None:
        print("=" * 60)
        print("🚀 Loading Z-Image-Turbo model...")
        print("=" * 60)
        print("📦 Model: Tongyi-MAI/Z-Image-Turbo (6B parameters)")
        print("💾 Size: ~12GB (will download on first run)")
        print("⏳ This may take a few minutes on first launch...")
        print("=" * 60)
        
        # Use bfloat16 for optimal performance on supported GPUs
        # trust_remote_code=True allows loading the custom ZImagePipeline from the model repo
        pipe = DiffusionPipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=True,  # Required for custom Z-Image pipeline
        )
        
        print("=" * 60)
        print("✅ Model loaded successfully!")
        print("🔄 Moving model to CUDA...")
        pipe.to("cuda")
        print("✅ Model ready on GPU!")
        print("=" * 60)
        
        # Enable Flash Attention for better efficiency (optional)
        try:
            # Check if flash-attn is installed
            import flash_attn
            pipe.transformer.set_attention_backend("flash")
            print("✓ Flash Attention enabled successfully!")
        except ImportError:
            print("⚠ Flash Attention not installed, using default SDPA attention")
            print("  (This is fine - the model will still work great!)")
        except Exception as e:
            print(f"⚠ Flash Attention not available: {e}")
            print("  Falling back to default SDPA attention")
        
        # [Optional] Model Compilation
        # Compiling the DiT model accelerates inference, but the first run will take longer to compile.
        # pipe.transformer.compile()
        
        # [Optional] CPU Offloading
        # Enable CPU offloading for memory-constrained devices.
        # pipe.enable_model_cpu_offload()
        
        print("Model loaded successfully!")
    return pipe

def generate_image(
    prompt,
    negative_prompt,
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
        print("\n" + "🔄 Initializing generation...")
        pipeline = load_model()
        
        # Handle seed
        if use_random_seed:
            seed = np.random.randint(0, 2**32 - 1)
        
        generator = torch.Generator("cuda").manual_seed(int(seed))
        
        # Generate image
        print(f"🎨 Generating image...")
        print(f"   Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"   Resolution: {width}x{height}")
        print(f"   Steps: {num_inference_steps} (= {num_inference_steps-1} DiT forwards)")
        print(f"   Seed: {seed}")
        
        image = pipeline(
            prompt=prompt,
            height=int(height),
            width=int(width),
            num_inference_steps=int(num_inference_steps),  # This actually results in 8 DiT forwards when set to 9
            guidance_scale=float(guidance_scale),           # Guidance should be 0 for the Turbo models
            generator=generator,
        ).images[0]
        
        print("✅ Image generated successfully!\n")
        return image, f"✅ Generated successfully!\n🎲 Seed: {seed}\n📐 Resolution: {width}x{height}\n⚡ Steps: {num_inference_steps}"
    
    except Exception as e:
        error_msg = f"❌ Error generating image: {str(e)}"
        print(error_msg + "\n")
        # Return a blank image and error message
        blank_image = Image.new('RGB', (512, 512), color='gray')
        return blank_image, error_msg

# Example prompts
example_prompts = [
    [
        "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.",
        "",
        1024,
        1024,
        9,
        0.0,
        42,
        False
    ],
    [
        "A serene mountain landscape at sunset, photorealistic, 8k, detailed",
        "blurry, low quality, distorted",
        1024,
        1024,
        9,
        0.0,
        123,
        False
    ],
    [
        "A futuristic city with flying cars, cyberpunk style, neon lights, detailed architecture",
        "",
        1024,
        768,
        9,
        0.0,
        456,
        False
    ],
    [
        "A golden cat playing in a garden, sunlight, flowers, photorealistic",
        "",
        768,
        768,
        9,
        0.0,
        789,
        False
    ],
    [
        "Portrait of a wizard with a long white beard, magical staff, fantasy art, detailed",
        "ugly, deformed, blurry",
        1024,
        1024,
        9,
        0.0,
        101,
        False
    ],
]

# Create Gradio interface
with gr.Blocks(title="Z-Image-Turbo Generator") as demo:
    gr.Markdown(
        """
        # ⚡️ Z-Image-Turbo Image Generator
        
        Generate high-quality images with **Z-Image-Turbo** - an efficient 6B parameter model with sub-second inference!
        
        **Features:**
        - 📸 **Photorealistic Quality**: Strong photorealistic image generation with excellent aesthetic quality
        - 📖 **Accurate Bilingual Text Rendering**: Excels at rendering complex Chinese and English text
        - ⚡️ **Ultra-Fast**: Only 8 inference steps needed (set num_inference_steps to 9)
        - 🎨 **Creative Generation**: Powered by Single-Stream Diffusion Transformer (S3-DiT) architecture
        - 🚀 **Decoupled-DMD**: Advanced acceleration algorithm for optimal performance
        
        **Important:** For Turbo models, keep guidance_scale at 0.0 for best results.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input controls
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want to generate...",
                lines=5,
                value="Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern."
            )
            
            negative_prompt = gr.Textbox(
                label="Negative Prompt (Optional)",
                placeholder="What you don't want in the image...",
                lines=2
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
                    label="Inference Steps (9 = 8 DiT forwards)"
                )
                guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=0.0,
                    label="Guidance Scale (Keep at 0.0 for Turbo)"
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
            
            generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")
            
            gr.Markdown("### 📝 Tips for Best Results:")
            gr.Markdown(
                """
                - **Be Specific**: Detailed prompts produce better results
                - **Optimal Settings**: Use 1024x1024 resolution, 9 inference steps, guidance_scale 0.0
                - **Style Keywords**: Include art style, lighting, mood descriptors
                - **Text Rendering**: Model excels at Chinese and English text in images
                - **Inference Steps**: 9 steps = 8 DiT forwards (optimal for Turbo)
                - **Guidance Scale**: Must be 0.0 for Turbo models
                """
            )
        
        with gr.Column(scale=1):
            # Output
            output_image = gr.Image(
                label="Generated Image",
                type="pil",
                height=600
            )
            output_info = gr.Textbox(
                label="Generation Info",
                lines=2
            )
            
            gr.Markdown("### 💾 Save Options:")
            with gr.Row():
                download_btn = gr.Button("📥 Download Image")
    
    # Examples section
    gr.Markdown("## 🎨 Example Prompts")
    gr.Examples(
        examples=example_prompts,
        inputs=[
            prompt,
            negative_prompt,
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
        label="Click on an example to load it"
    )
    
    # Connect the generate button
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
            use_random_seed
        ],
        outputs=[output_image, output_info]
    )
    
    gr.Markdown(
        """
        ---
        ### 📚 About Z-Image-Turbo
        
        **Z-Image** is an efficient image generation foundation model with **Single-Stream Diffusion Transformer** architecture.
        
        **Z-Image-Turbo** is a distilled version that matches or exceeds leading competitors with only **8 NFE** (Number of Function Evaluations).
        
        **Key Technologies:**
        - 🏗️ **S3-DiT Architecture**: Scalable Single-Stream Diffusion Transformer
        - ⚡️ **Decoupled-DMD**: Core acceleration algorithm enabling 8-step generation
        - 🤖 **DMDR**: Fusing DMD with Reinforcement Learning for enhanced performance
        - 📊 **6B Parameters**: Optimal balance of quality and efficiency
        
        **Performance:**
        - Sub-second inference latency on enterprise GPUs
        - State-of-the-art results on AI Arena leaderboard
        - Photorealistic quality with bilingual text rendering
        
        **Links:**
        - 🌐 [ModelScope](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo)
        - 🤗 [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
        """
    )

# Launch the app
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎨 Z-Image-Turbo Gradio UI")
    print("=" * 60)
    print("📍 Starting server...")
    print("🌐 Server will be available at: http://localhost:7860")
    print("=" * 60)
    print("ℹ️  Note: Model will download (~12GB) on first image generation")
    print("=" * 60 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

