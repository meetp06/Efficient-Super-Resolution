import os
import numpy as np
import time
from PIL import Image
import statistics
import maccel  # NPU Driver
import glob

# --- SETTINGS ---
# 1. Priority: Look for the FAST model first
if os.path.exists("super_resolution_fast.mxq"):
    model_path = "super_resolution_fast.mxq"
    print("🚀 Testing FAST Model: super_resolution_fast.mxq")
else:
    model_path = "super_resolution.mxq"
    print("⚠️ Fast model not found. Testing STANDARD model: super_resolution.mxq")

# Find directories automatically if possible, else use defaults
lr_dir = next((d for d in ["test_images", "LR_val"] if os.path.exists(d)), "LR_val")
hr_dir = next((d for d in ["HR_val"] if os.path.exists(d)), "HR_val")

print(f"📂 Input Dir: {lr_dir}")
print(f"📂 Target Dir: {hr_dir}")

# --- HELPER: PSNR ---
def calculate_psnr(img1, img2):
    # Ensure both are in range [0, 1]
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 10 * np.log10(1.0 / mse)

# --- 1. LOAD IMAGES ---
print("⏳ Loading images...")
test_images = []
hr_images = []
filenames = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg'))])

for img_file in filenames:
    # 1. Load LR (Input) - USE BICUBIC TO MATCH TRAINING!
    img = Image.open(os.path.join(lr_dir, img_file)).convert('RGB').resize((256, 256), Image.BICUBIC)
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # 2. Transpose to NCHW (Batch, Channel, Height, Width)
    # This matches the ONNX export format
    img_nchw = np.expand_dims(img_array.transpose(2, 0, 1), 0).astype(np.float32)
    test_images.append(img_nchw)
    
    # 3. Load HR (Target)
    hr_path = os.path.join(hr_dir, img_file)
    if os.path.exists(hr_path):
        hr = Image.open(hr_path).convert('RGB').resize((256, 256), Image.BICUBIC)
        hr_array = np.array(hr, dtype=np.float32) / 255.0
        hr_images.append(hr_array) # Keep as HWC for PSNR math
    else:
        hr_images.append(None)

print(f"✅ Loaded {len(test_images)} pairs.")

# --- 2. NPU INITIALIZATION ---
print("🚀 Initializing NPU...")
try:
    acc = maccel.Accelerator(0)
    model = maccel.Model(model_path)
    model.launch(acc)
    print("✅ Model Launched!")
except Exception as e:
    print(f"❌ Init Failed: {e}")
    exit()

# --- 3. INFERENCE LOOP ---
print("⚡ Running Benchmark...")
latencies = []
psnr_scores = []

# Warmup
for _ in range(5):
    model.infer([test_images[0]])

for i, img in enumerate(test_images):
    start = time.perf_counter()
    
    # INFERENCE
    outputs = model.infer([img])
    
    end = time.perf_counter()
    latencies.append((end - start) * 1000)

    # Calculate Score
    if hr_images[i] is not None:
        # Get output tensor
        out_tensor = outputs[0] 
        
        # Handle dimensions: (1, 3, 256, 256) -> (256, 256, 3)
        out_img = np.squeeze(out_tensor) # (3, 256, 256)
        out_img = np.transpose(out_img, (1, 2, 0)) # (256, 256, 3)
        
        score = calculate_psnr(out_img, hr_images[i])
        psnr_scores.append(score)

# --- 4. REPORT ---
avg_latency = statistics.mean(latencies)
avg_psnr = statistics.mean(psnr_scores) if psnr_scores else 0.0

print("\n" + "="*30)
print(f"📊 FINAL SCORECARD")
print(f"   Model:   {model_path}")
print(f"   Images:  {len(latencies)}")
print(f"   Runtime: {avg_latency:.4f} ms")
print(f"   PSNR:    {avg_psnr:.4f} dB")
print("="*30)

model.dispose()