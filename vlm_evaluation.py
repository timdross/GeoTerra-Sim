import pandas as pd
import json
import os
import io
import base64
import concurrent.futures
from PIL import Image
from openai import OpenAI

# 1. Configuration
model_node_ip = os.getenv("MODEL_HOST", "127.0.0.1") 
client = OpenAI(
    base_url=f"http://example_node:8000/v1",
    api_key="local-hpc-no-key-needed"
)

LOCAL_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

DATASET_BASE_DIR = "/path/to/img_dem_dir"
ORIGINAL_CSV_PATH = "img_dem_dataset.csv"
UNIVERSAL_OVERLAP_CSV = "vlm_evaluation_reference.csv"
RESULTS_OUTPUT_PATH = "zero_shot_evaluation_results.csv"

# 2. Rubric & Prompting
SYSTEM_PROMPT = """
You are an expert geospatial analyst and remote sensing validator.
You will be provided with a target text prompt and a generated multi-modal image. 
The image consists of two vertical panels:
- TOP PANEL: A generated 1-meter resolution RGB aerial scene (NAIP).
- BOTTOM PANEL: The corresponding generated elevation heightmap (DEM).

==================================================
STEP 1: VISUAL ATTRIBUTE VERIFICATION (Image Reliance Mitigation)
==================================================
Before scoring, you must explicitly evaluate the following physical characteristics of the image panels. Answer each with a clear "Yes" or "No" and provide a brief visual observation:
1. NADIR PERSPECTIVE: Does the top panel depict a true top-down (nadir) aerial view?
2. TARGET TEXTURES: Are the specific NLCD features requested in the target prompt visually detectable as distinct textures in the top panel? Do not list textures that were not requested.
3. COHERENCE: Do the physical features shown in the top panel align spatially with the contours and elevation gradients in the bottom panel?

==================================================
STEP 2: DECOMPOSED SCORING RUBRICS (Multidimensional Fidelity)
==================================================
Before generating your final scores, you must independently analyze the spatial layout of the imagery. 
CRITICAL CONSTRAINTS: 
- Do NOT repeat the target prompt.
- Do NOT use the names of the biome or ecoregion in your analysis.
- You must strictly answer the structural questions outlined in the JSON schema below.

DIMENSION A: SEMANTIC PROMPT ALIGNMENT (1-5)
- Score 1: Completely fails to match the prompt (e.g., shows an entirely incorrect biome or land cover type).
- Score 2: Shows an approximate match to the target biome, but fails to reflect any of the specific requested NLCD textures or object distributions.
- Score 3: Exhibits characteristics associated with the correct biome, but contains moderate visual artifacts or unrealistic land cover characteristics given the ecoregion (e.g., vegetation uniformly or artificially placed).
- Score 4: Highly accurate alignment with the target biome and textures, but with minor feature distribution inaccuracies or a slight color palette that does not perfectly reflect the specific target ecoregion nuances.
- Score 5: Flawless alignment; perfect representation of the specific biome, exact and natural land cover/object distribution, and correct color palette reflecting the target ecoregion description.

DIMENSION B: VISUAL QUALITY & GEOSPATIAL FIDELITY (1-5)
- Score 1: Completely fails basic aerial photography standards (e.g., shows a non-aerial perspective or uninterpretable visual fields).
- Score 2: Correct aerial perspective, but the imagery is incredibly blurry, unrealistically monochromatic, or demonstrates entirely incorrect 1-meter resolution scaling.
- Score 3: Distinct terrain features are visible, but contains moderate visual artifacts, rendering noise, or unrealistic surface textures that degrade geospatial fidelity.
- Score 4: Highly accurate aerial scene with sharp 1-meter resolution fidelity, but with minor visual artifacts or slight pixel blurs that do not fully degrade realism.
- Score 5: Flawless visual quality; perfect 1-meter nadir aerial perspective, accurate and highly crisp surface textures, and excellent edge definitions.

DIMENSION C: CROSS-MODAL COHERENCE (1-5)
- Score 1: Complete cross-modal failure; surface features in the top panel completely contradict or misalign with the underlying 3D topography of the bottom panel (e.g., flat water mapped over a steep mountain ridge).
- Score 2: Correct perspective in both panels, but the imagery demonstrates completely incorrect resolution scaling between modalities, or the elevation gradients show severe misalignment relative to the surface objects.
- Score 3: General structural features correlate at a macro level, but contain moderate cross-modal discrepancies or terrain characteristics distributed uniformly without respecting natural topographic slopes.
- Score 4: Highly accurate geometric and structural alignment between the NAIP visual features and the DEM contours, with only minor spatial shifts or minor terrain boundary inaccuracies.
- Score 5: Flawless cross-modal coherence; perfect spatial and structural correlation where high-elevation landmarks map exactly to corresponding visible features and low-elevation contours naturally host correct ecosystems.

==================================================
OUTPUT FORMAT
==================================================
You must output valid JSON strictly matching this schema:
{
    "attribute_verification": {
        "is_nadir_view": "Yes/No - [observation]",
        "prominent_textures_present": "Yes/No - [observation]",
        "geometric_alignment_present": "Yes/No - [observation]"
    },
    "chain_of_thought": {
        "1_surface_textures": "Describe exactly what surface textures are visible in the top-left, center, and bottom-right of the top panel.",
        "2_elevation_points": "Identify exactly where the highest (brightest) and lowest (darkest) elevation points are located in the bottom DEM panel.",
        "3_geographic_synthesis": "Do the surface textures identified in step 1 geographically make sense with the elevation gradients identified in step 2? (e.g., water in valleys, rocks on ridges)."
    },
    "semantic_alignment_score": <int 1-5>,
    "visual_quality_score": <int 1-5>,
    "cross_modal_coherence_score": <int 1-5>
}
"""


# 3. Image Processing Helper

def get_stacked_base64(naip_path, dem_path):
    """Stacks the generated NAIP and DEM vertically, resizes, and encodes to base64."""
    img_naip = Image.open(naip_path).convert("RGB")
    img_dem = Image.open(dem_path).convert("RGB")
    
    w, h = img_naip.size
    stacked = Image.new("RGB", (w, h * 2))
    stacked.paste(img_naip, (0, 0))
    stacked.paste(img_dem, (0, h))
    
    # SPEED OPTIMIZATION: Shrink by 50% to massively reduce vLLM token processing
    stacked = stacked.resize((w // 2, h), Image.Resampling.LANCZOS)
    
    buffered = io.BytesIO()
    stacked.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# 4. Core Evaluation Worker

def process_single_evaluation(args):
    """Worker that evaluates a specific experiment's outputs for a given image_id."""
    row, experiment_name, naip_path, dem_path = args
    image_id = row['image_id']
    
    if not isinstance(naip_path, str) or not os.path.exists(naip_path) or not os.path.exists(dem_path):
        return None # Skip if files are missing on disk

    # 1. Reconstruct the target conditioning prompt from the merged metadata
    biome = row['biome']
    eco_name = row['eco_name']
    prompt_parts = [f"A 1-meter resolution aerial scene featuring a {biome} biome within the {eco_name} ecoregion."]
    
    rel_path = str(row['path']).lstrip('/')
    meta_path = os.path.join(DATASET_BASE_DIR, rel_path, "metadata.json")
    
    try:
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            
        nlcd = metadata.get("NLCD", {})
        prominent_features = [k for k, v in nlcd.items() if float(v) > 0.25]
        
        if prominent_features:
            if len(prominent_features) > 1:
                features_str = ", ".join(prominent_features[:-1]) + f", and {prominent_features[-1]}"
            else:
                features_str = prominent_features[0]
            prompt_parts.append(f"The terrain is heavily characterized by {features_str}.")
    except Exception:
        pass # Proceed with base prompt if metadata fails
        
    conditioning_prompt = " ".join(prompt_parts)
    
    # 2. Encode the generated images
   # base64_image = get_stacked_base64(naip_path, dem_path)

    # 3. Execute Zero-Shot CoT API Call
    try:
        base64_image = get_stacked_base64(naip_path, dem_path)
        response = client.chat.completions.create(
            model=LOCAL_MODEL_NAME,
            response_format={"type": "json_object"},
            temperature=0.1, 
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": f"TARGET PROMPT: \"{conditioning_prompt}\""},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
        )
        
        result_json = json.loads(response.choices[0].message.content)
        
        return {
            "image_id": image_id,
            "experiment": experiment_name,
            "target_prompt": conditioning_prompt,
            "attribute_verification": json.dumps(result_json.get("attribute_verification", {})),
            "chain_of_thought": json.dumps(result_json.get("chain_of_thought", {})),
            "semantic_alignment": result_json.get("semantic_alignment_score", 0),
	    "visual_quality": result_json.get("visual_quality_score", 0),
            "cross_modal_coherence": result_json.get("cross_modal_coherence_score", 0)
        }
        
    except Exception as e:
        print(f"Error evaluating {image_id} [{experiment_name}]: {e}")
        return None


# 5. Thread Pool Execution & Data Prep

def evaluate_dataset(max_workers=16):
    print(f"Loading {UNIVERSAL_OVERLAP_CSV}...")
    univ_df = pd.read_csv(UNIVERSAL_OVERLAP_CSV)
    
    print(f"Loading original metadata from {ORIGINAL_CSV_PATH}...")
    orig_df = pd.read_csv(ORIGINAL_CSV_PATH)
    orig_df['image_id'] = orig_df['path'].apply(lambda x: os.path.basename(os.path.normpath(str(x))))
    
    merged_df = pd.merge(univ_df, orig_df, on='image_id', how='left')
    experiment_names = [c.replace('_naip_path', '') for c in univ_df.columns if c.endswith('_naip_path')]
    
    # --- CHECKPOINT: Load completed evaluations ---
    evaluations = []
    completed_keys = set()
    
    if os.path.exists(RESULTS_OUTPUT_PATH):
        try:
            existing_df = pd.read_csv(RESULTS_OUTPUT_PATH)
            if not existing_df.empty:
                evaluations = existing_df.to_dict(orient='records')
                # Create unique keys of (image_id, experiment) that are already completed
                completed_keys = set(zip(existing_df['image_id'], existing_df['experiment']))
                print(f"Found existing checkpoint! Loaded {len(completed_keys)} already completed evaluations.")
        except Exception as e:
            print(f"Could not read existing checkpoint file, starting fresh: {e}")

    # Build the task queue, skipping already evaluated items
    tasks = []
    skipped_count = 0
    for _, row in merged_df.iterrows():
        for exp in experiment_names:
            image_id = row['image_id']
            if (image_id, exp) in completed_keys:
                skipped_count += 1
                continue
            
            naip_col = f"{exp}_naip_path"
            dem_col = f"{exp}_dem_path"
            tasks.append((row, exp, row[naip_col], row[dem_col]))
            
    print(f"Skipped {skipped_count} completed tasks. Queueing remaining {len(tasks)} tasks.")

    if len(tasks) == 0:
        print("All evaluations are already completed!")
        print_summary(pd.DataFrame(evaluations))
        return

    # Execute the queue incrementally
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to get futures mapped to their calls
        future_to_task = {
            executor.submit(process_single_evaluation, task): task 
            for task in tasks
        }
        
        try:
            # as_completed yields futures as soon as they finish
            for count, future in enumerate(concurrent.futures.as_completed(future_to_task), 1):
                result = future.result()
                if result:
                    evaluations.append(result)
                    
                    # Periodically save to disk (every 10 runs to prevent disk write bottlenecks)
                    if count % 10 == 0 or count == len(tasks):
                        pd.DataFrame(evaluations).to_csv(RESULTS_OUTPUT_PATH, index=False)
                        print(f"Progress Saved: {len(evaluations)} total evaluations written to disk.")
                        
        except KeyboardInterrupt:
            print("\nEvaluation paused by user. Saving current progress...")
            pd.DataFrame(evaluations).to_csv(RESULTS_OUTPUT_PATH, index=False)
            print("Checkpoint saved successfully. You can safely resume later.")
            return

    # Final Save
    results_df = pd.DataFrame(evaluations)
    results_df.to_csv(RESULTS_OUTPUT_PATH, index=False)
    print_summary(results_df)


def print_summary(results_df):
    """Helper to display metrics summary."""
    if len(results_df) > 0:
        print("\n" + "="*50)
        print("✅ CURRENT SUMMARY METRICS")
        print("="*50)
        
        for metric in ["semantic_alignment", "visual_quality", "cross_modal_coherence"]:
            if metric in results_df.columns:
                mean_scores = results_df.groupby('experiment')[metric].mean()
                print(f"\nMean Score for {metric.replace('_', ' ').title()}:")
                for exp, score in mean_scores.items():
                    print(f" - {exp:<20}: {score:.2f} / 5.0")
    else:
        print("No completed records found to generate metrics.")

if __name__ == "__main__":
    evaluate_dataset(max_workers=16)
