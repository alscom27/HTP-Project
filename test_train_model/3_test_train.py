from ultralytics import YOLO

# 스크립트가 직접 실행될 때만 아래 코드가 실행되도록 감싸줍니다.
if __name__ == "__main__":
    model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv8 model

    # Train the model on a custom dataset
    model.train(
        data="test_data/data.yaml", epochs=50, imgsz=640, batch=16, device="0"
    )  # Adjust parameters as needed
