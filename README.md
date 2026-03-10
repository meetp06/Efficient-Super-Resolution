# Efficient Super Resolution on NPU

This project demonstrates an end-to-end pipeline for deploying an efficient Super Resolution model on a Neural Processing Unit (NPU). It converts a standard ONNX super-resolution model into an NPU-compatible MXQ format to achieve maximum inference speed, and includes benchmarking scripts to evaluate both performance (latency) and image quality (PSNR).

## Features
- **Model Conversion**: Converts an ONNX model to MXQ using the `qubee` compiler with a speed-optimized quantization strategy (Per-Tensor Max Quantization).
- **Calibration**: Generates calibration data from a set of images to aid the quantization process.
- **NPU Inference**: Uses the `maccel` driver to execute the compiled model on hardware accelerators.
- **Benchmarking**: Measures and reports processing latency (ms) and calculates the Peak Signal-to-Noise Ratio (PSNR) compared to high-resolution targets.

## Project Structure
- `step_2_onnx_to_mxq.py`: Script to prepare calibration data and compile the ONNX model into a fast MXQ model.
- `step_3_inference.py`: Script that initializes the NPU, runs the compiled model against a set of input images, and evaluates PSNR/latency.
- `training_script_Meetkumar_Patel.ipynb`: Jupyter notebook detailing the initial training or setup of the super-resolution model.

## Requirements
- Python 3.x
- NumPy, Pillow
- `qubee` (for model compilation and calibration)
- `maccel` (NPU Driver for inference)