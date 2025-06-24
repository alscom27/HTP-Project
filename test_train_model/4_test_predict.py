from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")  # Load a pretrained YOLOv8 model

pred = model.predict(
    source="test_data/images/train/tree_7_female_00068.jpg",  # Path to the test images
    conf=0.8,  # Confidence threshold
    save=True,  # Save predictions
)


print("Prediction results:")
for result in pred:
    print(f"Image: {result.path}")
    for box in result.boxes:
        print(f"  Class: {box.cls}, Confidence: {box.conf:.2f}, BBox: {box.xyxy}")
