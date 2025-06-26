import cvat_sdk.auto_annotation as cvataa
from ultralytics import YOLO

# 모델 로드
model = YOLO(
    r"C:\Users\main\Desktop\HTP-Project\runs\detect\yolo12n_all_data\weights\best.pt"
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


# pip install "cvat-sdk==2.39.0" "cvat-cli==2.39.0"

# cvat-cli --server-host http://192.168.0.204:8082/ --auth kimkva123@gmail.com:Hegikh324985 task auto-annotate 42 --function-file test_train_model/9_auto_labeling.py --allow-unmatched-labels --clear-existing
# cvat-cli --server-host http://192.168.0.204:8082/ --auth kimkva123@gmail.com:Hegikh324985
