
# YOLO to OpenVINO Conversion Guide

This repository provides a guide and tools for converting YOLO models (e.g., YOLOv8) into OpenVINO's Intermediate Representation (IR) format for optimized inference on Intel hardware, including CPU and GPU.

## System Requirements

Before starting, ensure your system meets the following requirements from the [OpenVINO documentation](https://docs.openvino.ai/2024/about-openvino/release-notes-openvino/system-requirements.html#operating-systems-and-developer-environment):

### Operating System:
- Windows 10 or higher.

### Required Software:
- **Microsoft Visual Studio 2019** with the following components:
  - **MSVC v142 - VS 2019 C++ x64/x86 build tools**
  - **Windows 10 SDK**
  - **C++ CMake tools for Windows** (if available)
- **CMake** version 3.16 or higher.
- **Python** version 3.8-3.12 (future versions are likely supported as well).
- **Intel® HD Graphics Driver** for inference on GPU (optional but recommended for GPU support).

## Installation Steps

1. **Install Required Libraries:**
   After setting up your environment, install the necessary libraries:
   
   ```bash
   pip install ultralytics
   pip install openvino
   pip install openvino-dev
   ```

   Optionally, install any additional dependencies using the requirements file:

   ```bash
   pip install -r requirements.txt
   ```

2. **Convert YOLO to ONNX:**
   Use the provided `yolo_to_onnx.py` script to convert your YOLOv8 model to ONNX format:

   ```bash
   python yolo_to_onnx.py
   ```

3. **Convert ONNX to OpenVINO IR (XML/BIN):**
   Once the ONNX model is created, navigate to the folder containing your ONNX file and run the following command to convert it to OpenVINO IR format:

   ```bash
   python -m openvino.tools.mo --input_model plate.onnx --input_shape [1,3,640,640]
   ```

   This will generate the `.xml` and `.bin` files necessary for inference with OpenVINO.

4. **Running Inference:**
   To run inference on a video file, modify the `Vedio_xml_start.py` script. Ensure that you provide the correct paths to the `.xml` file and the video (e.g., `video.mp4`), then run:

   ```bash
   python Vedio_xml_start.py
   ```

Now, sit back and enjoy the results as the YOLO model runs using OpenVINO-optimized inference!
