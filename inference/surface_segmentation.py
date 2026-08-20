import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ----- Configuration -----

MODEL_DIR = "/path/to/model"
OUTPUT_DIR = "./inference_results"
IMAGE_PATH = "/path/to/test/image"

id2label = {0: "background", 1: "water", 2: "tree canopy", 3: "low vegetation", 4: "barren"}
NUM_CLASSES = len(id2label)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- Initialization -----

processor = SegformerImageProcessor.from_pretrained(MODEL_DIR)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# ----- Processing + Inference -----

# Load image as RGB
image_pil = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.array(image_pil)

# Inference
with torch.no_grad():
    encoded_inputs = processor(images=image_np, return_tensors="pt")
    pixel_values = encoded_inputs["pixel_values"].to(device)

    outputs = model(pixel_values=pixel_values)
    
    # Upsample to match original image size (768, 768)
    logits = torch.nn.functional.interpolate(
        outputs.logits, 
        size=image_np.shape[:2], 
        mode="bilinear", 
        align_corners=False
    )
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

# ----- Save Visualization -----

colors = ['black', 'blue', 'darkgreen', 'lightgreen', 'red']
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

# 1. Get exact dimensions
height, width = pred.shape
dpi = 100  # Sets the pixel scale factor

# 2. Set figure size explicitly based on pixel dimensions
plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

# 3. Ensure the image fills the entire axes area
ax = plt.axes([0, 0, 1, 1])  
ax.set_axis_off()

plt.imshow(pred, cmap=cmap, norm=norm)

# 4. Save without 'tight' layout, as we've already handled the margins
plt.savefig(
    os.path.join(OUTPUT_DIR, "prediction.png"), 
    dpi=dpi, 
    pad_inches=0
)
plt.close()

print(f"Prediction saved to {OUTPUT_DIR}/prediction.png")
