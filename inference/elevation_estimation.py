import os
import json
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from datetime import datetime
from tqdm import tqdm

# Import the original dataset class from your training script
from train_height_diff import OptimizedElevationDataset, ElevationRegressionHead

def main():
    csv_file = "img_dem_dataset.csv"
    base_dir = "/path/to/img_dem_dir"
    pretrained_model_dir = "/MESA_weights"
    weights_path = "saved_heads/elevation_regression_head.pt"
    stats_path = "saved_heads/elevation_stats.json"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Initialization ---")
    print(f"Running Inference Evaluation on: {device}")

    # ----- Load Tokenizer and Text Encoder -----
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_dir, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_dir, subfolder="text_encoder").to(device)

    # ----- Initialize the Test Dataset -----
    try:
        test_dataset = OptimizedElevationDataset(
            csv_file=csv_file, base_dir=base_dir, 
            tokenizer=tokenizer, text_encoder=text_encoder, 
            device=device, split="test"
        )
    except RuntimeError as e:
        print(f"\n[ERROR] Failed to initialize test split: {e}")
        print("Make sure 'saved_heads/elevation_stats.json' exists from a prior training run.")
        return

    # Load raw stats for denormalization tracking
    with open(stats_path, "r") as f:
        stats = json.load(f)
    target_mean = stats["mean"][0]
    target_std = stats["std"][0]

    # ----- Clean up text encoder to save VRAM -----
    del text_encoder
    torch.cuda.empty_cache()

    # ----- Set up DataLoader for the test set -----
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # ----- Load the trained regression head -----
    mlp = ElevationRegressionHead(input_dim=1024).to(device)
    if os.path.exists(weights_path):
        mlp.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded trained model weights from {weights_path}")
    else:
        print(f"[WARNING] Weights not found at {weights_path}. Running with random initialization!")
    mlp.eval()

    # ----- Evaluation Loop & Ground Truth Comparison -----
    print("\n" + "="*50)
    print(f"Evaluating {len(test_dataloader)} Samples from Test Split")
    print("="*50)

    total_absolute_error = 0.0
    total_relative_error = 0.0
    num_samples_to_print = 5 

    with torch.no_grad():
        for idx, (embeddings, normalized_targets) in enumerate(test_dataloader):
            embeddings = embeddings.to(device, non_blocking=True)
            normalized_targets = normalized_targets.to(device, non_blocking=True)

            # Model prediction (normalized space)
            normalized_preds = mlp(embeddings)

            # Convert back to true height metrics (Meters)
            pred_meters = max(0.0, (normalized_preds.item() * target_std) + target_mean)
            true_meters = (normalized_targets.item() * target_std) + target_mean

            # Calculate Errors
            absolute_error = abs(pred_meters - true_meters)
            total_absolute_error += absolute_error
            relative_error = absolute_error / max(true_meters, 1e-4)
            total_relative_error += relative_error

            # Print detailed context for the first few samples
            if idx < num_samples_to_print:
                raw_row = test_dataset.raw_rows.iloc[idx]
                rel_path = str(raw_row['path']).lstrip('/')
                
                meta_path = os.path.join(base_dir, rel_path, "metadata.json")
                try:
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    
                    biome = str(raw_row['biome']) if pd.notna(raw_row['biome']) else None
                    eco_name = str(raw_row['eco_name']) if pd.notna(raw_row['eco_name']) else None
                    
                    date_str = metadata.get("VIS", {}).get("date", str(raw_row['naip_date']))
                    if pd.isna(raw_row['naip_date']) and not metadata.get("VIS", {}).get("date"):
                        month = "an unspecified month" 
                    else:
                        clean_date = date_str[:10].replace("/", "-")
                        month = datetime.strptime(clean_date, "%Y-%m-%d").strftime("%B")
                    
                    prompt_parts = ["A 1-meter resolution aerial scene"]
                    if biome: prompt_parts.append(f"featuring a {biome} biome")
                    if eco_name: prompt_parts.append(f"within the {eco_name} ecoregion")
                    prompt_parts.append(f"during {month}")
                    
                    text_prompt = " ".join(prompt_parts) + "."
                    
                    # Reconstruct the NLCD Auxiliary Sentence
                    nlcd = metadata.get("NLCD", {})
                    prominent = [k for k, v in nlcd.items() if float(v) > 0.25]
                    if prominent:
                        feat_str = prominent[0] if len(prominent) == 1 else ", ".join(prominent[:-1]) + f", and {prominent[-1]}"
                        text_prompt += f" The terrain is heavily characterized by {feat_str}."
                except Exception:
                    text_prompt = "[Failed to load metadata.json for prompt reconstruction]"
                
                print(f"\n[Sample {idx + 1}] Path: {rel_path}")
                print(f"  -> Full Input Prompt:   {text_prompt}")
                print(f"  -> Ground Truth Relief: {true_meters:.2f} meters")
                print(f"  -> Predicted Relief:    {pred_meters:.2f} meters")
                print(f"  -> Absolute Error:      {absolute_error:.2f} meters")
                
                display_rel_error = min(relative_error * 100, 999.99)
                percent_suffix = "+" if display_rel_error == 999.99 else "%"
                print(f"  -> Relative Error:      {display_rel_error:.2f}{percent_suffix}")

        # Compute Final Metrics
        mae = total_absolute_error / len(test_dataloader)
        mrae = (total_relative_error / len(test_dataloader)) * 100
        
        print("\n" + "="*50)
        print(f"Overall Test Split Performance Metrics:")
        print(f"  -> Total Evaluation Samples: {len(test_dataloader)}")
        print(f"  -> Mean Absolute Error (MAE):           {mae:.2f} meters")
        print(f"  -> Mean Relative Absolute Error (MRAE): {mrae:.2f}%")
        print("="*50)

if __name__ == "__main__":
    main()