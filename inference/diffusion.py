import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  # Added for DEM visualization mapping
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

# Import your custom architecture
from models import UNetDEMConditionModel

@torch.no_grad()
def main():
    # ----- Configuration & Setup -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    weight_dtype = torch.bfloat16
    
    base_model_id = "Manojb/stable-diffusion-2-1-base"
    
    # Point this to the specific epoch folder you want to generate from
    checkpoint_dir = "/path/to/trained/model/checkpoint"
    output_dir = "./inference_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Prompt matching your ablation format
    prompt = "A 1-meter resolution aerial scene featuring a Tropical & Subtropical Grasslands Savannas & Shrublands biome within the Western Gulf coastal grasslands ecoregion. The terrain is heavily characterized by cultivated."

    num_inference_steps = 50
    guidance_scale = 7.5
    batch_size = 1

    print("Loading frozen text encoder and VAE...")
    tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder").to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae").to(device, dtype=weight_dtype)
    
    # We use DDIM for fast high-quality inference
    scheduler = DDIMScheduler.from_pretrained(
        base_model_id, 
        subfolder="scheduler",
        prediction_type="v_prediction",
        clip_sample=False
    )

    # Encode ONLY the text prompt (Delete the uncond_inputs block completely)
    text_inputs = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

    print(f"Loading custom dual-head MESA U-Net from {checkpoint_dir}...")
    unet = UNetDEMConditionModel.from_pretrained(
        checkpoint_dir, 
        torch_dtype=weight_dtype
    ).to(device)
    unet.eval()

    # ----- Textual Encoding -----
    print("Encoding prompts...")
    text_inputs = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]

    uncond_inputs = tokenizer(
        [""] * batch_size, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]

    text_embeddings = torch.cat([uncond_embeds, prompt_embeds])

    # ----- Initialization -----
    scheduler.set_timesteps(num_inference_steps, device=device)
    
    # Initialize random latent distribution (B, 8, 96, 96)
    latents = torch.randn(
        (batch_size, 8, 96, 96), 
        device=device, 
        dtype=weight_dtype
    )
    latents = latents * scheduler.init_noise_sigma

    # ----- The Denoising Loop -----
    print("Beginning Denoising Loop...")
    for i, t in enumerate(scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Pass the prompt embeddings directly, do not concatenate with empty embeddings
        noise_pred = unet(
            sample=latents,
            timestep=t,
            encoder_hidden_states=prompt_embeds, # Single prompt only
            timestep_cond=None 
        ).sample

        # Step directly without CFG math
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # ----- Decoding the Modalities -----
    print("Decoding latents via VAE...")
    latents_img, latents_dem = torch.split(latents, [4, 4], dim=1)

    # Decode NAIP Image
    latents_img = 1 / vae.config.scaling_factor * latents_img
    img_output = vae.decode(latents_img).sample
    
    # Decode DEM Elevation
    latents_dem = 1 / vae.config.scaling_factor * latents_dem
    dem_output = vae.decode(latents_dem).sample

    # ----- Post-Processing & Saving -----
    # Process RGB NAIP Image
    img_output = (img_output / 2 + 0.5).clamp(0, 1)
    img_array = img_output.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    img_array = (img_array * 255).round().astype(np.uint8)
    
    pil_img = Image.fromarray(img_array)
    pil_img.save(os.path.join(output_dir, "generated_naip.png"))
    print("Saved generated NAIP image to: generated_naip.png")

    # Process DEM Elevation Mapping
    dem_array = dem_output.cpu().float().numpy()[0]
    dem_1d = np.mean(dem_array, axis=0)  # This is strictly 768x768

    # Save the raw data array
    np.save(os.path.join(output_dir, "generated_dem.npy"), dem_1d)

    # Rescale the DEM to 0-255
    dem_min, dem_max = dem_1d.min(), dem_1d.max()
    dem_scaled = 255 * (dem_1d - dem_min) / (dem_max - dem_min + 1e-8)
    dem_uint8 = dem_scaled.astype(np.uint8)

    pil_dem = Image.fromarray(dem_uint8)
    pil_dem.save(os.path.join(output_dir, "generated_dem_visualization.png"))
    print("Saved true 768x768 DEM topographic PNG via PIL.")
    print("Saved visual DEM topographic png to: generated_dem_visualization.png")
    
    print("Inference Complete!")

if __name__ == "__main__":
    main()
