import os
import json
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datetime import datetime

class NAIPDEMDataset(Dataset):
    def __init__(self, csv_file: str, base_dir: str = "/scratch/tdross/768x768_NAIP_DEM/files", split: str = "train", ablation_prompt_eng: bool = False):
        """
        Args:
            csv_file (str): Path to the sample_points.csv file containing the split column.
            base_dir (str): Root directory of the dataset structure.
            split (str): The specific split to load ('train', 'val', or 'test').
            ablation_prompt_eng (bool): If True, appends prominent NLCD features to the text prompt.
        """
        self.ablation_prompt_eng = ablation_prompt_eng
        
        # Load the CSV file
        all_data = pd.read_csv(csv_file)
        
        # Force split to lower case and strip white space to avoid string matching mismatches
        all_data['split'] = all_data['split'].astype(str).str.lower().str.strip()
        target_split = split.lower().strip()
        
        # Filter the dataframe to only include rows corresponding to the target split
        self.data = all_data[all_data['split'] == target_split].reset_index(drop=True)
        
        if len(self.data) == 0:
            raise ValueError(f"No samples found for split '{split}' in {csv_file}. Check your data formatting.")
            
        print(f"Loaded {len(self.data)} samples successfully for the '{target_split}' split pipeline.")
        
        self.base_dir = base_dir
        
        self.naip_transform = transforms.Compose([
            transforms.Resize((768, 768)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            # 1. Fetch paths and row data
            row = self.data.iloc[idx]
            rel_path = str(row['path']).lstrip('/') 
            sample_dir = os.path.join(self.base_dir, rel_path)
            
            naip_path = os.path.join(sample_dir, "naip.png")
            meta_path = os.path.join(sample_dir, "metadata.json")
            dem_npy_path = os.path.join(sample_dir, "dem.npy")

            if not (os.path.exists(naip_path) and os.path.exists(meta_path) and os.path.exists(dem_npy_path)):
                raise FileNotFoundError(f"Missing files in {sample_dir}")

            # 2. Load and Process NAIP Image
            naip_img = Image.open(naip_path).convert("RGB")
            naip_tensor = self.naip_transform(naip_img)

            # 3. Load and Process metadata.json
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            # 4. Extract Textual Data & Format Base Template
            biome = str(row['biome']) if pd.notna(row['biome']) else "unknown"
            eco_name = str(row['eco_name']) if pd.notna(row['eco_name']) else "unknown"
            
            date_str = metadata.get("VIS", {}).get("date", str(row['naip_date']))
            clean_date = date_str[:10].replace("/", "-") 
            month = datetime.strptime(clean_date, "%Y-%m-%d").strftime("%B")
            
            text_prompt = f"A 1-meter resolution aerial scene featuring a {biome} biome within the {eco_name} ecoregion."

            # ==========================================
            # ABLATION 1: Prompt Engineering Formatting
            # ==========================================
            nlcd = metadata.get("NLCD", {})
            if self.ablation_prompt_eng:
                # Find all land cover types that represent more than 25% of the image
                prominent_features = [k for k, v in nlcd.items() if float(v) > 0.25]
                if prominent_features:
                    if len(prominent_features) > 1:
                        features_str = ", ".join(prominent_features[:-1]) + f", and {prominent_features[-1]}"
                    else:
                        features_str = prominent_features[0]
                    text_prompt += f" The terrain is heavily characterized by {features_str}."

            # 5. Extract Scalar Data
            dem_max_global = metadata.get("DEM", {}).get("max", 0.0)
            dem_min_global = metadata.get("DEM", {}).get("min", 0.0)
            dem_range = (dem_max_global - dem_min_global) / 5000.0
            
            scalars = [
                dem_range,
                nlcd.get("water", 0.0), nlcd.get("snow", 0.0), nlcd.get("barren", 0.0),
                nlcd.get("forest", 0.0), nlcd.get("shrub", 0.0),
                nlcd.get("herbaceous", nlcd.get("heraceous", 0.0)),
                nlcd.get("cultivated", 0.0), nlcd.get("wetland", 0.0)
            ]
            scalar_tensor = torch.tensor(scalars, dtype=torch.float32)

            # 6. NaN-Targeted DEM Loading & Normalization
            dem_data = np.load(dem_npy_path).astype(np.float32).squeeze()
            valid_mask = (~np.isnan(dem_data)).astype(np.float32)
            
            if np.any(valid_mask):
                local_min = np.nanmin(dem_data)
                local_max = np.nanmax(dem_data)
                local_range = local_max - local_min
                
                dem_clean = np.nan_to_num(dem_data, nan=local_min)

                if local_range > 0:
                    dem_normalized = 2.0 * ((dem_clean - local_min) / local_range) - 1.0
                else:
                    dem_normalized = np.zeros_like(dem_clean)
            else:
                dem_normalized = np.zeros_like(dem_data)
                dem_normalized = np.nan_to_num(dem_normalized, nan=0.0)

            # 7. Convert to Tensors
            dem_tensor = torch.from_numpy(dem_normalized).float().unsqueeze(0)
            dem_tensor = dem_tensor.repeat(3, 1, 1) 
            
            mask_tensor = torch.from_numpy(valid_mask).float().unsqueeze(0).unsqueeze(0)
            latent_mask = torch.nn.functional.interpolate(mask_tensor, size=(96, 96), mode='area').squeeze(0)
            latent_mask = (latent_mask > 0.5).float()

            return {
                "image": naip_tensor,
                "dem": dem_tensor,
                "dem_mask": latent_mask,
                "metadata": scalar_tensor,
                "text": text_prompt
            }

        except Exception as e:
            # Pick a completely random index and try again to maintain the batch size
            random_idx = np.random.randint(0, len(self.data))
            return self.__getitem__(random_idx)
