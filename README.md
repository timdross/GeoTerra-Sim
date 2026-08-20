# GeoTerra-Sim

### Installing weights from HuggingFace (https://huggingface.co/rosst1221/geoterra-sim)

We provide the final weights for the following models used within the GeoTerra-Sim paper:
- diffusion finetuning
  - baseline
  - ablation: metadata embedding
  - ablation: loss scaling
- surface segmentation
- height estimation

Download the pre-trained diffusion weights into your local directory structure:

```bash
# Create the target directory
mkdir -p weights/diffusion

# Download model checkpoints from HuggingFace
hf download rosst1221/geoterra-sim --include "baseline/*" --local-dir weights/diffusion/baseline
hf download rosst1221/geoterra-sim --include "metadata/*" --local-dir weights/diffusion/metadata
hf download rosst1221/geoterra-sim --include "scale/*"    --local-dir weights/diffusion/scale
```

To download the pretrained MESA weights:
```bash
# Create the target directory
mkdir -p weights/mesa

# Download MESA model weights from HuggingFace
hf download NewtNewt/MESA --local-dir weights/mesa
```

To download the surface segmentation model:
```bash
# Create the target directory
mkdir -p weights/segmentation

# Download surface segmentation model weights from HuggingFace
hf download rosst1221/geoterra-sim --include "segmentation/*"    --local-dir weights/surface_segmentation
```
