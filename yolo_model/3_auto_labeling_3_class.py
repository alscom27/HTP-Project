import cvat_sdk.auto_annotation as cvataa
from ultralytics import YOLO

# 모델 로드
model = YOLO(
    r"C:\Users\main\Desktop\HTP-Project\runs\detect\yolo11n_all_box_all_data\weights\best.pt"
)

# 라벨 사양 정의
spec = cvataa.DetectionFunctionSpec(
    labels=[
        cvataa.label_spec("house", 0, type="rectangle"),
        cvataa.label_spec("person", 1, type="rectangle"),
        cvataa.label_spec("tree", 2, type="rectangle"),
    ],
)


# 감지 함수 정의
def detect(context, image):
    results = model(image)
    detections = []
    for result in results:
        for box in result.boxes:
            label_index = int(box.cls)
            label_name = spec.labels[label_index].name
            detections.append(
                cvataa.rectangle(
                    label_id=label_index,
                    points=box.xyxy.tolist()[0],
                    score=float(box.conf),
                )
            )
    return detections