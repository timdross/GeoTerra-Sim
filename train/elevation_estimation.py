import os
import json
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ----- Beefed-Up Regression Architecture -----
class ElevationRegressionHead(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        # Deeper network with GELU activations for better gradient flow
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1) 
        )

    def forward(self, x):
        return self.net(x)

# ----- High-Performance Pre-cached Dataset -----
class OptimizedElevationDataset(Dataset):
    def __init__(self, csv_file: str, base_dir: str, tokenizer, text_encoder, device, split: str = "train"):
        all_data = pd.read_csv(csv_file)
        all_data['split'] = all_data['split'].astype(str).str.lower().str.strip()
        target_split = split.lower().strip()
        
        self.raw_rows = all_data[all_data['split'] == target_split].reset_index(drop=True)
        self.base_dir = base_dir
        
        if len(self.raw_rows) == 0:
            raise ValueError(f"No samples found for split '{split}' in {csv_file}.")
            
        print(f"Pre-loading {len(self.raw_rows)} metadata profiles using multi-threading...")
        
        raw_samples = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(
                executor.map(self._load_single_row, [row for _, row in self.raw_rows.iterrows()]),
                total=len(self.raw_rows),
                desc="Reading JSONs"
            ))
        
        raw_samples = [r for r in results if r is not None]

        print(f"Pre-computing and caching CLIP embeddings to RAM...")
        self.embeddings = []
        self.targets = []

        batch_size = 256
        text_encoder.eval()
        
        with torch.no_grad():
            for i in tqdm(range(0, len(raw_samples), batch_size), desc="Encoding Text"):
                batch = raw_samples[i:i+batch_size]
                texts = [b["text"] for b in batch]
                targets = [b["target"] for b in batch]

                inputs = tokenizer(texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
                outputs = text_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                embeddings = outputs.pooler_output.cpu().float()

                self.embeddings.append(embeddings)
                self.targets.append(torch.stack(targets))

        self.embeddings = torch.cat(self.embeddings, dim=0)
        self.targets = torch.cat(self.targets, dim=0)

        if split == "train":
            self.target_mean = self.targets.mean(dim=0)
            self.target_std = self.targets.std(dim=0)
            self.target_std[self.target_std == 0] = 1.0 
            
            print(f"\n--- Training Normalization Stats ---")
            print(f"Mean (Min/Max): {self.target_mean.tolist()}")
            print(f"Std (Min/Max):  {self.target_std.tolist()}\n")

            os.makedirs("saved_heads", exist_ok=True)
            stats_dict = {
                "mean": self.target_mean.tolist(),
                "std": self.target_std.tolist()
            }
            with open("saved_heads/elevation_stats.json", "w") as f:
                json.dump(stats_dict, f, indent=4)
        else:
            # Load the training stats from disk for the val/test sets
            try:
                with open("saved_heads/elevation_stats.json", "r") as f:
                    stats_dict = json.load(f)
                self.target_mean = torch.tensor(stats_dict["mean"])
                self.target_std = torch.tensor(stats_dict["std"])
            except FileNotFoundError:
                raise RuntimeError("Training stats not found! You must initialize the 'train' split before the 'val' split.")

        # Scale targets down to [-1, 1] range for stable training
        self.targets = (self.targets - self.target_mean) / self.target_std

        # Save stats to disk for later inference
        os.makedirs("saved_heads", exist_ok=True)
        stats_dict = {
            "mean": self.target_mean.tolist(),
            "std": self.target_std.tolist()
        }
        with open("saved_heads/elevation_stats.json", "w") as f:
            json.dump(stats_dict, f, indent=4)
        print("Saved normalization stats to saved_heads/elevation_stats.json")


    def _load_single_row(self, row):
        try:
            rel_path = str(row['path']).lstrip('/') 
            meta_path = os.path.join(self.base_dir, rel_path, "metadata.json")

            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            dem_max = metadata.get("DEM", {}).get("max", 0.0)
            dem_min = metadata.get("DEM", {}).get("min", 0.0)

            elevation_range = abs(dem_max - dem_min)
            target_tensor = torch.tensor([elevation_range], dtype=torch.float32)

            biome = str(row['biome']) if pd.notna(row['biome']) else None
            eco_name = str(row['eco_name']) if pd.notna(row['eco_name']) else None
            
            date_str = metadata.get("VIS", {}).get("date", str(row['naip_date']))
            clean_date = date_str[:10].replace("/", "-")
            month = datetime.strptime(clean_date, "%Y-%m-%d").strftime("%B")
            
            prompt_parts = ["A 1-meter resolution aerial scene"]
            if biome: prompt_parts.append(f"featuring a {biome} biome")
            if eco_name: prompt_parts.append(f"within the {eco_name} ecoregion")
            prompt_parts.append(f"during {month}")

            text_prompt = " ".join(prompt_parts) + "."

            nlcd = metadata.get("NLCD", {})
            prominent_features = [k for k, v in nlcd.items() if float(v) > 0.25]
            if prominent_features:
                if len(prominent_features) > 1:
                    features_str = ", ".join(prominent_features[:-1]) + f", and {prominent_features[-1]}"
                else:
                    features_str = prominent_features[0]
                text_prompt += f" The terrain is heavily characterized by {features_str}."

            return {"text": text_prompt, "target": target_tensor}
        except Exception:
            return None

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]

# ----- Streamlined Training Loop -----
def main():
    csv_file = "img_dem_dataset.csv"
    base_dir = "/path/to/img_dem_dir"
    pretrained_model_dir = "/MESA_weights" # we use the textual encoder fromo MESA
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Optimized Training on: {device}")

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_dir, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_dir, subfolder="text_encoder").to(device)

    # Initialize Train (calculates and saves stats)
    train_dataset = OptimizedElevationDataset(
        csv_file=csv_file, base_dir=base_dir, 
        tokenizer=tokenizer, text_encoder=text_encoder, 
        device=device, split="train"
    )
    
    # Initialize Val (loads train stats)
    val_dataset = OptimizedElevationDataset(
        csv_file=csv_file, base_dir=base_dir, 
        tokenizer=tokenizer, text_encoder=text_encoder, 
        device=device, split="val"
    )
    
    del text_encoder
    torch.cuda.empty_cache()

    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    mlp = ElevationRegressionHead(input_dim=1024).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.HuberLoss() 
    
    # Scheduler now monitors Validation Loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Early Stopping Setup
    epochs = 100 
    early_stopping_patience = 10
    epochs_without_improvement = 0
    best_val_loss = float('inf')
    save_path = "saved_heads/elevation_regression_head.pt"

    for epoch in range(epochs):

        # TRAINING PHASE
        mlp.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for embeddings, targets in progress_bar:
            embeddings = embeddings.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            preds = mlp(embeddings)
            loss = criterion(preds, targets)
            
            optimizer.zero_grad(set_to_none=True) 
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"train_huber": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_dataloader)

        # VALIDATION PHASE
        mlp.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for embeddings, targets in val_dataloader:
                embeddings = embeddings.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                preds = mlp(embeddings)
                loss = criterion(preds, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

        # CHECKPOINT & EARLY STOPPING
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(mlp.state_dict(), save_path)
            print(f"  -> Validation improved! Saved best model to {save_path}")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement. Patience: {epochs_without_improvement}/{early_stopping_patience}")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs! Best Val Loss: {best_val_loss:.4f}")
            break

if __name__ == "__main__":
    main()
