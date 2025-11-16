from ultralytics import YOLO
import json
import os
from pathlib import Path

import pydicom
import numpy as np
from PIL import Image


import os
from pathlib import Path



dir="../final_testset/BT_TestSet2/"
save_dir="../final_testset/pngVersion2/"

isdicom=False



# Change this to your test data directory
read_dir=save_dir  # Update this path to your test data
output_json_path = "./predictions.json"

model = YOLO("..\\training_code\\runs\\classify\\yolo11m-cls_v3_aug2\\weights\\best.pt")




def read_dicom(path):

    ds = pydicom.dcmread(path)
    # 2. Convert to pixel array
    arr=ds.pixel_array
    # 3. (Optional) Apply rescale slope/intercept
    slope = getattr(ds, 'RescaleSlope', 1.0)
    intercept = getattr(ds, 'RescaleIntercept', 0.0)
    arr = arr * slope + intercept

    # 4. Windowing (convert to 0–255)
    #    If the DICOM has WindowCenter/Width tags, use them:
    if ('WindowCenter' in ds) and ('WindowWidth' in ds):
        center = ds.WindowCenter
        width = ds.WindowWidth
        # Handle multi-valued tags
        if isinstance(center, pydicom.multival.MultiValue):
            center = center[0]
            width = width[0]
        low = center - width/2
        high = center + width/2
        arr = np.clip(arr, low, high)
    arr = (arr - arr.min()) / (arr.max() - arr.min())  # normalize 0–1
    arr = (arr * 255).astype(np.uint8)                  # scale to 0–255
    return arr

IMAGE_EXTENSIONS = {'.dcm'}

def read_images(root_dir):
    root = Path(root_dir)
    for path in root.rglob('*'):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            img = read_dicom(path)
            yield path, img

#pip install GDCM, pylibjpeg 
#pip install pylibjpeg-libjpeg
#pip install pylibjpeg-openjpeg
#these will install numpy 2.x and it broke my environment
#so be careful

if(isdicom):
    for path, img in read_images(dir):
        if(os.path.exists(save_dir) == False):
            os.makedirs(save_dir, exist_ok=True)
            print(f"Directory created: {save_dir}")
        labeledDir=os.path.join(save_dir, path.parent.stem)
        os.makedirs(labeledDir, exist_ok=True)

        img = Image.fromarray(img)
        img.save(os.path.join(labeledDir, str(path.name)[:-4]+"_"+str(path.parent.stem) + ".png"))
        print(f"{path}: size={img.size}")











IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

def read_images(root_dir):
    root = Path(root_dir)
    for path in root.rglob('*'):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path

def map_prediction_to_labels(pred_label):
    """Map prediction labels to the required JSON format labels"""
    label_mapping = {
        "var": {"stroke": 1, "stroke_type": 3},
        "yok": {"stroke": 0, "stroke_type": 3},
    }
    return label_mapping.get(pred_label, {"stroke": 0, "stroke_type": 0})

# Initialize the JSON structure
predictions_json = {
    "kunye": {
        "takim_adi": "MAJESTYT",
        "takim_id": "742174",
        "aciklama": "CT Tahmin Verileri",
        "versiyon": "v1.0"
    },
    "tahminler": []
}

# Process images and generate predictions
for path in read_images(read_dir):
    if "MR" in str(path):
        continue

    # Extract filename without extension
    filename = path.stem + path.suffix
    
    result = model.predict(source=path, save=False)
    result = result[0]
    pred = result.probs.top1
    predlabel = result.names[pred]

    print(f"File: {filename}, Predicted: {predlabel}")
    # Map prediction to required format and add to JSON
    prediction_labels = map_prediction_to_labels(predlabel)
    
    prediction_entry = {
        "filename": filename,
        "stroke": prediction_labels["stroke"],
        "stroke_type": prediction_labels["stroke_type"]
    }
    
    predictions_json["tahminler"].append(prediction_entry)


os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
# Save predictions to JSON file
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(predictions_json, f, ensure_ascii=False, indent=2)

print(f"\nPredictions saved to {output_json_path}")
print(f"Total predictions: {len(predictions_json['tahminler'])}")