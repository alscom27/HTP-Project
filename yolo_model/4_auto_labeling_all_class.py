import cvat_sdk.auto_annotation as cvataa
from ultralytics import YOLO

# --- 1. 신뢰도 임계값 설정 ---
# 이 값을 조정하여 필터링할 신뢰도 점수를 변경할 수 있습니다. (예: 0.7로 하면 70% 이상)
CONF_THRESHOLD = 0.7

# 모델 로드
model = YOLO(r"best.pt")

# 라벨 사양 정의
spec = cvataa.DetectionFunctionSpec(
    labels=[
        cvataa.label_spec("house", 0, type="rectangle"),
        cvataa.label_spec("tree", 1, type="rectangle"),
        cvataa.label_spec("person", 2, type="rectangle"),
        cvataa.label_spec("door", 3, type="rectangle"),
        cvataa.label_spec("roof", 4, type="rectangle"),
        cvataa.label_spec("window", 5, type="rectangle"),
        cvataa.label_spec("branch", 6, type="rectangle"),
        cvataa.label_spec("root", 7, type="rectangle"),
        cvataa.label_spec("crown", 8, type="rectangle"),
        cvataa.label_spec("fruit", 9, type="rectangle"),
        cvataa.label_spec("gnarl", 10, type="rectangle"),
        cvataa.label_spec("eye", 11, type="rectangle"),
        cvataa.label_spec("leg", 12, type="rectangle"),
        cvataa.label_spec("mouth", 13, type="rectangle"),
        cvataa.label_spec("arm", 14, type="rectangle"),
    ],
)


# 감지 함수 정의
def detect(context, image):
    results = model(image)
    detections = []
    for result in results:
        for box in result.boxes:
            # --- 2. 신뢰도 값 확인 및 필터링 ---
            # box.conf 값이 설정한 CONF_THRESHOLD 값 이상일 때만 라벨을 추가합니다.
            if float(box.conf) >= CONF_THRESHOLD:
                label_index = int(box.cls)
                # spec.labels에 해당 인덱스가 있는지 안전하게 확인 (선택 사항이지만 권장)
                if label_index < len(spec.labels):
                    detections.append(
                        cvataa.rectangle(
                            label_id=label_index,
                            points=box.xyxy.tolist()[0],
                            score=float(box.conf),
                        )
                    )
    return detections


# cvat-cli --server-host http://192.168.0.204:8082/ --auth kimkva123@gmail.com:Hegikh324985 task auto-annotate 69 --function-file test_train_model/11_auto_labeling_all_box.py --allow-unmatched-labels --clear-existing
