from diffusers import AutoPipelineForText2Image
import torch

def generate_image(prompt: str, output_path, num_inference_steps=1, guidance_scale=0.0):
    """
    Generates an image using Stable Diffusion Turbo on Mac (MPS) or CPU.

    Args:
        prompt (str): The text prompt for image generation.
        output_path (str): File path to save the generated image. Default is "output.png".
        num_inference_steps (int): Number of inference steps. Default is 1.
        guidance_scale (float): Scale for classifier-free guidance. Default is 0.0.

    Returns:
        None
    """
    # Set device to MPS (Mac GPU) if available, otherwise fallback to CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load the model with appropriate dtype
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,  
        variant="fp16"
    )

    # Move model to device
    pipe.to(device)

    # Generate image
    image = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]

    # Save the image
    image.save(output_path)
    print(f"Image generated and saved at: {output_path}")

# Example usage:
save_image_to="output_stability.png"
generate_image("A cinematic shot of a baby dianasour wearing an intricate Italian priest robe.", output_path=save_image_to)
