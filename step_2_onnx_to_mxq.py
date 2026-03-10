import os
import numpy as np
from PIL import Image
import shutil
from qubee import mxq_compile
from qubee.calibration import make_calib_man

# --- PREPROCESSING ---
def preprocess_sr(image_path):
    # 1. Open and Resize
    img = Image.open(image_path).convert('RGB').resize((256, 256), Image.BICUBIC)
    
    # 2. Convert to 0-1 Float (HWC format for compiler)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array

# --- MAIN EXECUTION ---

# 1. Cleanup
if os.path.exists("calibration_data_sr"):
    shutil.rmtree("calibration_data_sr")

print("⚙️ Generating Calibration Data...")

# 2. Create Calibration Data
make_calib_man(
    pre_ftn=preprocess_sr,
    data_dir="calibration_images_npu", 
    save_dir=".",
    save_name="calibration_data_sr",
    max_size=len(os.listdir("calibration_images_npu"))
)

print("🚀 Compiling for MAXIMUM SPEED...")

# 3. Compile (Speed Configuration)
mxq_compile(
    model="super_resolution.onnx",
    calib_data_path="calibration_data_sr",
    
    # --- FIX IS HERE ---
    quantize_method="max",          # Changed from 'minmax' to 'max'
    is_quant_ch=False,              # Per-Tensor Quantization (Critical for Speed!)
    # -------------------
    
    quantize_percentile=0.9999,     # Ignored by max, but harmless
    topk_layer_name="",
    save_path="super_resolution_fast.mxq"
)

print("✅ DONE! Download 'super_resolution_fast.mxq' and run Step 3!")