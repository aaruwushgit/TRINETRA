"""
TensorRT Model Exporter for High-Throughput NVIDIA Edge Devices.

Converts YOLOv8 weights (.pt) to an FP16 TensorRT engine (.engine)
for 4x–6x inference speedup on NVIDIA Jetson / Tesla GPUs.

Usage:
  python scripts/export_tensorrt.py --weights yolov8n.pt --half
"""
import argparse
from pathlib import Path


def export_model(weights_path: str, half: bool = True):
    from ultralytics import YOLO

    path = Path(weights_path)
    if not path.exists():
        print(f"❌ Weights file not found: {weights_path}")
        return

    print(f"🚀 Exporting {weights_path} to TensorRT format (half={half})...")
    model = YOLO(weights_path)
    engine_path = model.export(format="engine", half=half, dynamic=False)
    print(f"✅ Successfully created TensorRT engine: {engine_path}")


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 to TensorRT")
    parser.add_argument("--weights", default="yolov8n.pt", help="Path to .pt weights file")
    parser.add_argument("--half", action="store_true", default=True, help="Enable FP16 precision")
    args = parser.parse_args()

    export_model(args.weights, half=args.half)


if __name__ == "__main__":
    main()
