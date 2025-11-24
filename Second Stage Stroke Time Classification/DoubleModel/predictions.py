from ultralytics import YOLO
import json
import os
from pathlib import Path

model = YOLO("../TrainingCodes/runs/classify/yolo11m-cls_datasetv2/weights/best.pt") 
HNmodel=YOLO("../TrainingCodes/runs/classify/yolo11m-cls_Hiperakut-Normalv2/weights/best.pt")
HSmodel=YOLO("../TrainingCodes/runs/classify/yolo11m-cls_Hiperakut-Subakutv2/weights/best.pt")
NSmodel=YOLO("../TrainingCodes/runs/classify/yolo11m-cls_Normal-Subakutv2/weights/best.pt")

# Change this to your test data directory
read_dir="./Yolo/Topluv2/val"  # Update this path to your test data
output_json_path = "./test_predictions.json"



IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

def read_images(root_dir):
    root = Path(root_dir)
    for path in root.rglob('*'):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path

def map_prediction_to_labels(pred_label):
    """Map prediction labels to the required JSON format labels"""
    label_mapping = {
        "HiperakutAkut": {"hyperacute_acute": 1, "subacute": 0, "normal_chronic": 0},
        "Subakut": {"hyperacute_acute": 0, "subacute": 1, "normal_chronic": 0},
        "NormalKronik": {"hyperacute_acute": 0, "subacute": 0, "normal_chronic": 1}
    }
    return label_mapping.get(pred_label, {"hyperacute_acute": 0, "subacute": 0, "normal_chronic": 0})

# Initialize the JSON structure
predictions_json = {
    "kunye": {
        "takim_adi": "MAJESTYT",
        "takim_id": "742174",
        "aciklama": "MR Tahmin Verileri",
        "versiyon": "v1.0"
    },
    "tahminler": []
}

# Process images and generate predictions
for path in read_images(read_dir):
    if "CT" in str(path):
        continue

    # Extract filename without extension
    filename = path.stem + path.suffix
    
    result = model.predict(source=path, save=False)
    result = result[0]
    pred = result.probs.top1
    predlabel = result.names[pred]
    conf = result.probs.top1conf
    newResult = None

    # Apply hybrid model logic if confidence is low
    if conf < 0.9:
        second_hit_label = result.names[result.probs.top5[1]]
        if (predlabel == "HiperakutAkut" and second_hit_label == "NormalKronik") or \
           (predlabel == "NormalKronik" and second_hit_label == "HiperakutAkut"):
            newResult = HNmodel.predict(source=path, save=False)
            newResult = newResult[0]
        elif (predlabel == "HiperakutAkut" and second_hit_label == "Subakut") or \
             (predlabel == "Subakut" and second_hit_label == "HiperakutAkut"):
            newResult = HSmodel.predict(source=path, save=False)
            newResult = newResult[0]
        elif (predlabel == "NormalKronik" and second_hit_label == "Subakut") or \
             (predlabel == "Subakut" and second_hit_label == "NormalKronik"):
            newResult = NSmodel.predict(source=path, save=False)
            newResult = newResult[0]

    # Get final prediction
    if newResult is None:
        final_pred_label = result.names[pred]
        print(f"File: {filename}, Predicted: {final_pred_label}")
    else:
        pred = newResult.probs.top1
        final_pred_label = newResult.names[pred]
        print(f"File: {filename}, Predicted: {final_pred_label} (hybrid model)")

    # Map prediction to required format and add to JSON
    prediction_labels = map_prediction_to_labels(final_pred_label)
    
    prediction_entry = {
        "filename": filename,
        "hyperacute_acute": prediction_labels["hyperacute_acute"],
        "subacute": prediction_labels["subacute"],
        "normal_chronic": prediction_labels["normal_chronic"]
    }
    
    predictions_json["tahminler"].append(prediction_entry)

# Save predictions to JSON file
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(predictions_json, f, ensure_ascii=False, indent=2)

print(f"\nPredictions saved to {output_json_path}")
print(f"Total predictions: {len(predictions_json['tahminler'])}")