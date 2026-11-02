import zipfile
from pathlib import Path
from ultralytics import YOLO

root = Path(r"G:\My Drive\datasets\Ai_Waste_Campaign")
checkpoint = root / "yolo_runs/yolo11n_waste_detection/weights/best.pt"
zip_path = root / "images.zip"
output = Path(r"D:\mlops\runs\existing_model_test")

with zipfile.ZipFile(zip_path) as archive:
    image_name = next(
        n for n in archive.namelist()
        if n.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    image_path = Path(r"D:\mlops\data") / Path(image_name).name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(archive.read(image_name))

model = YOLO(str(checkpoint))
print("Checkpoint:", checkpoint)
print("Model classes:", model.names)

results = model.predict(
    source=str(image_path),
    conf=0.25,
    save=True,
    project=str(output.parent),
    name=output.name,
    exist_ok=True
)

for result in results:
    print("Image:", image_path)
    print("Detections:", len(result.boxes))
    for box in result.boxes:
        cls = int(box.cls[0])
        print(model.names[cls], round(float(box.conf[0]), 3))

print("Inference output:", output)
