from ultralytics import YOLO

model = YOLO("best.pt")

if __name__ == "__main__":
    freeze_support = True  # Set to True to freeze the support layers
    model.val(
        data="test_data/4000_all_box_add_400_tree/data.yaml",
        imgsz=640,
        batch=16,
        device="0",
        name="yolo11n_4000_data_base",
    )  # Adjust parameters as needed
