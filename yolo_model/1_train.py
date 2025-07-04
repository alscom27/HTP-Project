from ultralytics import YOLO

# 스크립트가 직접 실행될 때만 아래 코드가 실행되도록 감싸줍니다.
if __name__ == "__main__":
    model = YOLO("yolo11n.pt")

    model.train(
        data="4000_all_box/data.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        device="0, 1",
        name="yolo11n_4000_data",
        patience=20,
    )  # Adjust parameters as needed
