import os
import csv
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, AutoencoderKL
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from dataset import NAIPDEMDataset
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file

from models import UNetDEMConditionModel

START_EPOCH = 0
EPOCHS_TO_RUN = 5

BATCH_SIZE = 8
NUM_WORKERS = 7
NUM_GPUS = 2
GRADIENT_ACCUM_STEPS = 8

ABLATION_PROMPT_ENG = True
ABLATION_LOSS_SCALING = True
RESUME_FROM_CHECKPOINT = True

RESUME_CHECKPOINT_DIR = "checkpoints/epoch_000"
DATASET_CSV = "img_dem_dataset.csv"

OUTPUT_METRICS_PATH = "/training_metrics.csv"

class MetadataEncoder(nn.Module):
    def __init__(self, input_dim=9, output_dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        # Output shape: [Batch, 1, 1024] so it can concatenate with text tokens
        return self.net(x)

# ==========================================
# 3. Main Training Loop
# ==========================================
def main():

    l_img_mult = 1.5 if ABLATION_LOSS_SCALING else 1.0
    l_dem_mult = 0.5 if ABLATION_LOSS_SCALING else 1.0

    print(f"LOSS SCALING: img-{l_img_mult}, dem-{l_dem_mult}")

    # Set this to the directory containing MESA's pretrained weights
    PRETRAINED_MODEL_DIR = "MESA_weights"

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
        kwargs_handlers=[ddp_kwargs]
    )
    device = accelerator.device
    weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(OUTPUT_METRICS_PATH), exist_ok=True)
        if not RESUME_FROM_CHECKPOINT or not os.path.exists(OUTPUT_METRICS_PATH):
            with open(OUTPUT_METRICS_PATH, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "avg_train_loss", "avg_val_loss"])

    # Load frozen components of MESA from pretrained directory
    print("Loading frozen text encoder, VAE, and scheduler from local checkpoint...")
    tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL_DIR, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_MODEL_DIR, subfolder="text_encoder").to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_DIR, subfolder="vae").to(device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    noise_scheduler = DDPMScheduler.from_pretrained(
        PRETRAINED_MODEL_DIR, 
        subfolder="scheduler",
        prediction_type="v_prediction",
        clip_sample=False
    )

    if not RESUME_FROM_CHECKPOINT:
        print("Initializing custom dual-head MESA U-Net...")
        # 4 channels are needed for Stable Diffusion 2.1
        unet = UNetDEMConditionModel(
            in_channels=4,
            out_channels=4,
            cross_attention_dim=1024,
            use_linear_projection=True
        )

    if not RESUME_FROM_CHECKPOINT:
        print("Loading fine-tuned MESA weights...")
        
        # Load the custom U-Net weights directly from the local safetensors file
        unet_weight_path = os.path.join(PRETRAINED_MODEL_DIR, "unet", "diffusion_pytorch_model.safetensors")
        state_dict = load_file(unet_weight_path)
        
        # Load weights with strict=False to bypass the unused conditioning projection layer
        missing_keys, unexpected_keys = unet.load_state_dict(state_dict, strict=False)

        # Filter out the known, acceptable missing key
        critical_missing = [k for k in missing_keys if "time_embedding.cond_proj" not in k]
        
        if critical_missing:
            print(f"WARNING: Found critical missing keys: {critical_missing}")
        if unexpected_keys:
            print(f"WARNING: Found unexpected keys in checkpoint: {unexpected_keys}")
        
        if not critical_missing and not unexpected_keys:
            print("MESA UNet weights mapped successfully (ignoring unused time_cond_proj).") 
    else:
        print("Loading from previous training...")
        
        # Hugging Face's from_pretrained handles the safetensors mapping automatically
        unet = UNetDEMConditionModel.from_pretrained(RESUME_CHECKPOINT_DIR)
        

    torch.cuda.empty_cache()
    unet.enable_gradient_checkpointing()

    # Data Loading...
    train_dataset = NAIPDEMDataset(csv_file=DATASET_CSV, split="train", ablation_prompt_eng=ABLATION_PROMPT_ENG)
    train_dataloader = DataLoader(train_dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    val_dataset = NAIPDEMDataset(csv_file=DATASET_CSV, split="val", ablation_prompt_eng=ABLATION_PROMPT_ENG)
    val_dataloader = DataLoader(val_dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    metadata_encoder = MetadataEncoder(input_dim=9, output_dim=1280).to(device, dtype=weight_dtype)

    if RESUME_FROM_CHECKPOINT and not ABLATION_PROMPT_ENG:
        meta_weight_path = os.path.join(RESUME_CHECKPOINT_DIR, "metadata_encoder.pth")
        if os.path.exists(meta_weight_path):
            print(f"Loading metadata encoder weights from {meta_weight_path}...")
            # Load the state dict and apply it to the instantiated model
            state_dict = torch.load(meta_weight_path, map_location=device, weights_only=True)
            metadata_encoder.load_state_dict(state_dict)
        else:
            print(f"WARNING: {meta_weight_path} not found. Encoder weights are randomly initialized.")
    
    # --- Optimizer ---
    # Since we are fine-tuning a model where the custom heads are already trained, 
    # we unify the learning rate to protect the established DEM/Image representations.
    if ABLATION_PROMPT_ENG:
        print("NOT ENCODING SCALAR METADATA")
        trainable_params = list(unet.parameters())
    else:
        trainable_params = list(unet.parameters()) + list(metadata_encoder.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=2e-5, weight_decay=1e-2)

    # --- Scheduler ---
    num_update_steps_per_epoch = len(train_dataloader) // accelerator.gradient_accumulation_steps
    
    TOTAL_EPOCHS = START_EPOCH + EPOCHS_TO_RUN
    max_train_steps = TOTAL_EPOCHS * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50, # Safely warm up to protect pre-trained weights
        num_training_steps=max_train_steps,
    )

    # Initialize accelerate
    unet, metadata_encoder, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        unet, metadata_encoder, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    if RESUME_FROM_CHECKPOINT:
        print(f"Restoring full training state from {RESUME_CHECKPOINT_DIR}...")
        accelerator.load_state(os.path.join(RESUME_CHECKPOINT_DIR, "accelerator_state"))

    if accelerator.is_main_process:
        print(f"DEBUG - Sample Prompt 1: {train_dataset[1]['text']}")
        print("Beginning Fine-Tuning on Local MESA Checkpoint...")

    # Training Loop
    for epoch in range(START_EPOCH, START_EPOCH + EPOCHS_TO_RUN):
        unet.train()
        epoch_train_loss_accum = 0.0

        for step, batch in enumerate(train_dataloader):
            if random.random() < 0.15:
                processed_texts = [""] * len(batch["text"])
            else:
                processed_texts = batch["text"]

            # 1. Encode Text Prompts
            text_inputs = tokenizer(
                processed_texts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            with torch.no_grad():
                encoder_hidden_states = text_encoder(text_inputs.input_ids.to(device))[0]
            
            if not ABLATION_PROMPT_ENG:
                meta_tensor = batch["metadata"].to(device, dtype=weight_dtype)

                if random.random() < 0.15:
                    meta_tensor = torch.zeros_like(meta_tensor)

                meta_timestep_cond = metadata_encoder(meta_tensor)
            else:
                meta_timestep_cond = None

            # 2. Encode Both Modalities to Latents
            with torch.no_grad():
                latents_img = vae.encode(batch["image"].to(device, dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                latents_dem = vae.encode(batch["dem"].to(device, dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor

            latents = torch.cat([latents_img, latents_dem], dim=1)

            # 3. Add Noise (Forward Diffusion)
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            target = noise_scheduler.get_velocity(latents, noise, timesteps)

            # 4. Forward Pass & Loss (Inside accumulation scope for distributed efficiency)
            with accelerator.accumulate(unet):
                # Predict Noise (MESA)
                model_pred = unet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep_cond=meta_timestep_cond
                ).sample

                # Split predictions and Compute Losses
                pred_img, pred_dem = torch.split(model_pred, [4, 4], dim=1)
                target_img, target_dem = torch.split(target, [4, 4], dim=1)
                loss_img = F.mse_loss(pred_img.float(), target_img.float(), reduction="mean") * l_img_mult

                dem_mask = batch["dem_mask"].to(device)
                unreduced_dem_loss = F.mse_loss(pred_dem.float(), target_dem.float(), reduction="none")
                masked_dem_loss = (unreduced_dem_loss * dem_mask).sum() / (dem_mask.sum() * pred_dem.shape[1] + 1e-8) * l_dem_mult

                total_loss = loss_img + masked_dem_loss

                # Backward Pass & Optimize
                accelerator.backward(total_loss)
                optimizer.step()
                lr_scheduler.step() # Advance the warmup scheduler
                optimizer.zero_grad() 

            gathered_train_loss = accelerator.gather(total_loss).mean()
            epoch_train_loss_accum += gathered_train_loss.item()

            global_step = step // accelerator.gradient_accumulation_steps

            # Print only when gradients sync AND we hit a multiple of 10 global steps
            if accelerator.is_main_process and accelerator.sync_gradients and global_step % 10 == 0:
                print(f"Epoch {epoch} | Global Step {global_step} | tot loss: {total_loss.item():.4f} | img loss: {loss_img.item():.4f} | dem loss: {masked_dem_loss.item():.4f}")

        # Calculate final average training loss for this epoch
        avg_train_loss = epoch_train_loss_accum / (step + 1)

        accelerator.wait_for_everyone()


        # Checkpoint Saving
        if accelerator.is_main_process:
            print(f"Saving weights and states for Epoch {epoch}...")
            save_dir = f"checkpoints/epoch_{epoch:03d}"
            os.makedirs(save_dir, exist_ok=True)

            # 1. Save the model weights (for inference/loading)
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_unet.save_pretrained(save_dir)

            unwrapped_meta_encoder = accelerator.unwrap_model(metadata_encoder)
            torch.save(
                unwrapped_meta_encoder.state_dict(), 
                os.path.join(save_dir, "metadata_encoder.pth")
            )           
            # 2. Save the full training state (optimizer, scheduler, RNG)
            accelerator.save_state(os.path.join(save_dir, "accelerator_state"))

        accelerator.wait_for_everyone()

        # Validation Loop
        unet.eval()
        val_loss_accum = 0.0
        
        if accelerator.is_main_process:
            print(f"--- Running Validation for Epoch {epoch} ---")

        with torch.no_grad():
            for val_step, val_batch in enumerate(val_dataloader):
                text_inputs = tokenizer(
                    val_batch["text"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                encoder_hidden_states = text_encoder(text_inputs.input_ids.to(device))[0]

                latents_img = vae.encode(val_batch["image"].to(device, dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                latents_dem = vae.encode(val_batch["dem"].to(device, dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                latents = torch.cat([latents_img, latents_dem], dim=1)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                target = noise_scheduler.get_velocity(latents, noise, timesteps)

                if not ABLATION_PROMPT_ENG:
                    meta_tensor = val_batch["metadata"].to(device, dtype=weight_dtype)
                    meta_timestep_cond = metadata_encoder(meta_tensor)
                else:
                    meta_timestep_cond = None

                model_pred = unet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep_cond=meta_timestep_cond
                ).sample

                pred_img, pred_dem = torch.split(model_pred, [4, 4], dim=1)
                target_img, target_dem = torch.split(target, [4, 4], dim=1)

                v_loss_img = F.mse_loss(pred_img.float(), target_img.float(), reduction="mean") * l_img_mult
                v_dem_mask = val_batch["dem_mask"].to(device)
                v_unreduced_dem_loss = F.mse_loss(pred_dem.float(), target_dem.float(), reduction="none")
                v_masked_dem_loss = (v_unreduced_dem_loss * v_dem_mask).sum() / (v_dem_mask.sum() * pred_dem.shape[1] + 1e-8) * l_dem_mult

                val_loss = v_loss_img + v_masked_dem_loss
                gathered_val_loss = accelerator.gather(val_loss).mean()
                val_loss_accum += gathered_val_loss.item()

        avg_val_loss = val_loss_accum / (val_step + 1)
        
        if accelerator.is_main_process:
            print(f">>> Epoch {epoch} Validation Complete | Avg Val Loss: {avg_val_loss:.4f} <<<")
            
            with open(OUTPUT_METRICS_PATH, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}"])

        accelerator.wait_for_everyone()

    accelerator.end_training()

if __name__ == "__main__":
    main()