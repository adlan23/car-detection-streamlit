import torch
from torch.serialization import add_safe_globals
from yolov5.models.yolo import DetectionModel


# Allow YOLOv5 model class
add_safe_globals({'DetectionModel': DetectionModel})

# File paths
input_path = "yolov5/runs/train/vehicle-detector/weights/vehicle_best.pt"
output_path = "yolov5/runs/train/vehicle-detector/weights/vehicle_best_linux.pt"

# Load full model with architecture (weights_only=False)
print(f"Loading: {input_path}")
model = torch.load(input_path, map_location="cpu", weights_only=False)

# Save only the weights (state_dict)
print(f"Saving: {output_path}")
torch.save(model['model'].state_dict(), output_path)

print("✅ Conversion complete.")

