import os
import random
import csv
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import (
    SegformerImageProcessor, 
    SegformerConfig, 
    SegformerForSemanticSegmentation,
    get_linear_schedule_with_warmup
)
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm
from accelerate import Accelerator

# ==========================================
# 1. Configuration & Taxonomy
# ==========================================
CHESAPEAKE_TRAIN_IMG = "/chesapeake_npy/train/images"
CHESAPEAKE_TRAIN_MASK = "/chesapeake_npy/train/masks"
CHESAPEAKE_VAL_IMG = "/chesapeake_npy/val/images"
CHESAPEAKE_VAL_MASK = "/chesapeake_npy/val/masks"

DEEPGLOBE_ROOT = "/DeepGlobe"
DEEPGLOBE_METADATA = "/DeepGlobe/metadata.csv"

id2label = {
    0: "background", 
    1: "water", 
    2: "tree canopy", 
    3: "low vegetation", 
    4: "barren"
}
label2id = {v: k for k, v in id2label.items()}
NUM_CLASSES = len(id2label)

MODEL_CHECKPOINT = "nvidia/segformer-b3-finetuned-ade-512-512"
BEST_MODEL_DIR = "./best_seg_model"

DEEPGLOBE_COLOR_MAP = {
    (0, 255, 255): 0,   # Urban -> Background
    (0, 0, 255): 1,     # Water -> Water
    (0, 255, 0): 2,     # Forest -> Tree canopy
    (255, 255, 0): 3,   # Agriculture -> Low vegetation
    (255, 0, 255): 3,   # Rangeland -> Low vegetation
    (255, 255, 255): 4, # Barren -> Barren
    (0, 0, 0): 0        # Unknown -> Background
}

# Training Hyperparameters
# note: effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
TOTAL_CPUS = 8
TOTAL_GPUS = 1
WORKERS_PER_GPU = 6
BATCH_SIZE = 8                
GRADIENT_ACCUMULATION_STEPS = 2 
LEARNING_RATE = 2e-4
EPOCHS = 100
PATIENCE = 10
TARGET_CROP_SIZE = 768

# Initialize Accelerator WITH gradient accumulation
accelerator = Accelerator(gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)

device = accelerator.device

print(f"Training on {device}")


# ==========================================
# 2. Dataset Classes
# ==========================================
class ChesapeakeDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # ----- Load image and normalize to uint8 -----
        image_npy = np.load(img_path)
        if image_npy.dtype != np.uint8 and image_npy.max() <= 1.0:
            image_npy = (image_npy * 255.0).astype(np.uint8)
        elif image_npy.dtype != np.uint8:
            image_npy = image_npy.astype(np.uint8)
            
        image = Image.fromarray(image_npy).convert("RGB")
        
        # ----- Load mask and drop trailing dimensions -----
        mask = np.load(mask_path)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        mask = mask.astype(np.uint8)
        mask_img = Image.fromarray(mask)

        # crop if needed
        w, h = image.size
        if w > TARGET_CROP_SIZE and h > TARGET_CROP_SIZE:
            x = random.randint(0, w - TARGET_CROP_SIZE)
            y = random.randint(0, h - TARGET_CROP_SIZE)
            image = image.crop((x, y, x + TARGET_CROP_SIZE, y + TARGET_CROP_SIZE))
            mask_img = mask_img.crop((x, y, x + TARGET_CROP_SIZE, y + TARGET_CROP_SIZE))
        else:
            # Fallback only if the NAIP tile is strangely tiny
            image = image.resize((TARGET_CROP_SIZE, TARGET_CROP_SIZE), Image.Resampling.BILINEAR)
            mask_img = mask_img.resize((TARGET_CROP_SIZE, TARGET_CROP_SIZE), Image.Resampling.NEAREST)

        # ----- DATA AUGMENTATION FLIPS -----
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask_img = mask_img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask_img = mask_img.transpose(Image.FLIP_TOP_BOTTOM)

        # ----- VECTORIZE MASKS -----
        mask_1d = np.array(mask_img)

        encoded = self.processor(images=image, segmentation_maps=mask_1d, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"].squeeze(0)
        }

class DeepGlobeDataset(Dataset):
    """Loads DeepGlobe images and visual RGB masks using a metadata CSV"""

    def __init__(self, root_dir, metadata_path, processor, split="train"):
        self.root_dir = root_dir
        self.processor = processor
        
        df = pd.read_csv(metadata_path)
        
        # ----- Pool both splits if they both contain valid images and masks -----
        valid_df = df[df['split'].isin(['train', 'val'])].sort_values(by='sat_image_path').reset_index(drop=True)
        
        np.random.seed(42)
        indices = np.random.permutation(len(valid_df))
        
        # 70% Train | 15% Val | 15% Test (Reserved for evaluation script)
        train_cutoff = int(len(valid_df) * 0.7)
        val_cutoff = int(len(valid_df) * 0.85)
        
        if split == "train":
            targeted_indices = indices[:train_cutoff]
        elif split == "val":
            targeted_indices = indices[train_cutoff:val_cutoff]
        elif split == "test":
            targeted_indices = indices[val_cutoff:]
            
        self.df = valid_df.iloc[targeted_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.root_dir, row['sat_image_path'])
        mask_path = os.path.join(self.root_dir, row['mask_path'])

        # ----- Load images -----
        image = Image.open(img_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("RGB")

        # ----- downsamples from 0.5m to 1.0m -----
        new_width = int(image.width * 0.5)
        new_height = int(image.height * 0.5)
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        mask_img = mask_img.resize((new_width, new_height), Image.Resampling.NEAREST)

        w, h = image.size
        if w > TARGET_CROP_SIZE and h > TARGET_CROP_SIZE:
            x = random.randint(0, w - TARGET_CROP_SIZE)
            y = random.randint(0, h - TARGET_CROP_SIZE)
            image = image.crop((x, y, x + TARGET_CROP_SIZE, y + TARGET_CROP_SIZE))
            mask_img = mask_img.crop((x, y, x + TARGET_CROP_SIZE, y + TARGET_CROP_SIZE))
        else:
            image = image.resize((TARGET_CROP_SIZE, TARGET_CROP_SIZE), Image.Resampling.BILINEAR)
            mask_img = mask_img.resize((TARGET_CROP_SIZE, TARGET_CROP_SIZE), Image.Resampling.NEAREST)

        # ----- Random Spatial Flips -----
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask_img = mask_img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask_img = mask_img.transpose(Image.FLIP_TOP_BOTTOM)

        mask_rgb = np.array(mask_img.convert("RGB"))  # <-- Add .convert("RGB") here
        mask_1d = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)

        for rgb, class_idx in DEEPGLOBE_COLOR_MAP.items():
            matches = np.all(mask_rgb == rgb, axis=-1)
            mask_1d[matches] = class_idx

        # ----- Pass to Processor -----
        encoded = self.processor(images=image, segmentation_maps=mask_1d, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"].squeeze(0)
        }

# ----- Initialization & Loaders -----

if accelerator.is_main_process:
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    csv_filename = "joint_training_metrics.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Val mIoU"])

processor = SegformerImageProcessor.from_pretrained(
    MODEL_CHECKPOINT, 
    do_reduce_labels=False,
    do_resize=False  
)

config = SegformerConfig.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_CLASSES,
    id2label=id2label,
    label2id=label2id
)

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_CHECKPOINT,
    config=config,
    ignore_mismatched_sizes=True
)

# Build Datasets
chesapeake_train = ChesapeakeDataset(CHESAPEAKE_TRAIN_IMG, CHESAPEAKE_TRAIN_MASK, processor)
deepglobe_train = DeepGlobeDataset(DEEPGLOBE_ROOT, DEEPGLOBE_METADATA, processor, split="train")
joint_train_dataset = ConcatDataset([chesapeake_train, deepglobe_train])

train_dataloader = DataLoader(
    joint_train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=WORKERS_PER_GPU, 
    pin_memory=True,
    persistent_workers=True
)


chesapeake_val = ChesapeakeDataset(CHESAPEAKE_VAL_IMG, CHESAPEAKE_VAL_MASK, processor)
deepglobe_val = DeepGlobeDataset(DEEPGLOBE_ROOT, DEEPGLOBE_METADATA, processor, split="val")
joint_val_dataset = ConcatDataset([chesapeake_val, deepglobe_val])

val_dataloader = DataLoader(
    joint_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=WORKERS_PER_GPU
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Setup Learning Rate Scheduler (Warmup for first 10% of training steps)
total_training_steps = (len(train_dataloader) * EPOCHS) // GRADIENT_ACCUMULATION_STEPS
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(total_training_steps * 0.1),
    num_training_steps=total_training_steps
)

metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=255).to(accelerator.device)

# Prepare everything with Accelerator
model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler
)

# ----- Training Loop -----

best_val_miou = 0.0
epochs_no_improve = 0

# These weights determine how heavily each class is penalized
weights = torch.tensor([ 5.0, 8.0, 1.2, 1.0, 10.0]).to(device)

loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0

    train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", disable=not accelerator.is_local_main_process)
    
    for step, batch in enumerate(train_pbar):
        # Context manager for gradient accumulation
        with accelerator.accumulate(model):
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            outputs = model(pixel_values=images)
            
            logits = torch.nn.functional.interpolate(
                outputs.logits, 
                size=labels.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )

            loss = loss_fn(logits, labels)
            
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        total_train_loss += loss.item()
        train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_train_loss = total_train_loss / len(train_dataloader)

    # --- Validation ---
    model.eval()
    metric.reset()
    
    val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", disable=not accelerator.is_local_main_process)
    with torch.no_grad():
        for batch in val_pbar:
            outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
            
            logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=batch["labels"].shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            preds = logits.argmax(dim=1)
            
            preds, labels = accelerator.gather_for_metrics((preds, batch["labels"]))
            metric.update(preds, labels)

    val_miou = metric.compute().item()

    # Wait for all GPUs to finish validation before doing I/O operations
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"\nEpoch {epoch+1} Results -> Train Loss: {avg_train_loss:.4f} | Validation mIoU: {val_miou:.4f}\n")
        
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_train_loss, val_miou])

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            epochs_no_improve = 0
            print(f"New Best mIoU: {best_val_miou:.4f}. Saving best model checkpoint...")
            
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(BEST_MODEL_DIR, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
            processor.save_pretrained(BEST_MODEL_DIR)
        else:
            epochs_no_improve += 1
            print(f"No improvement in mIoU for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= PATIENCE:
        if accelerator.is_main_process:
            print("Early stopping triggered. Training complete!")
        break
        
    # Free up memory at the end of the epoch
    torch.cuda.empty_cache()
