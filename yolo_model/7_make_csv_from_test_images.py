import os
import pandas as pd
import cv2
from ultralytics import YOLO


# -------------------------------------------------------------------
#                           분석 함수
# -------------------------------------------------------------------
def analyze_size_and_loc(image_w, image_h, box, group):
    """메인 객체의 크기와 위치를 분석하는 함수"""
    x_min, y_min, x_max, y_max = box
    box_center_x = (x_min + x_max) / 2
    left_boundary, right_boundary = image_w / 3, image_w * 2 / 3

    if box_center_x < left_boundary:
        location = "left"
    elif box_center_x > right_boundary:
        location = "right"
    else:
        location = "center"

    box_area = (x_max - x_min) * (y_max - y_min)
    image_area = image_w * image_h
    ratio = box_area / image_area if image_area > 0 else 0
    size = ""

    if group in ["house", "tree"]:
        if ratio >= 0.6:
            size = "big"
        elif ratio >= 0.16:
            size = "middle"
        else:
            size = "small"
    elif group == "person":
        if ratio >= 0.4:
            size = "big"
        elif ratio >= 0.16:
            size = "middle"
        else:
            size = "small"
    return size, location


# ===================================================================
#                           메인 스크립트
# ===================================================================

print("--- 스크립트 실행 시작 ---")

# 1.  사용자 설정 
BASE_PROJECT_DIR = r"C:\Users\main\Desktop\HTP-Project"
MODEL_PATH = os.path.join(BASE_PROJECT_DIR, "HTP_Detection.pt")
OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, "labeling_result")

# 분석할 폴더와 관련 정보 정의
TASKS = [
    {
        "name": "house",
        "path": os.path.join(BASE_PROJECT_DIR, "HTP_Datasets", "house", "test"),
        "main_obj_id": 0,
        "parts_map": {3: "door", 4: "roof", 5: "window"},
        "group_for_size_loc": "house",
    },
    {
        "name": "men",
        "path": os.path.join(BASE_PROJECT_DIR, "HTP_Datasets", "men", "test"),
        "main_obj_id": 2,
        "parts_map": {11: "eye", 12: "leg", 13: "mouth", 14: "arm"},
        "group_for_size_loc": "person",
    },
    {
        "name": "women",
        "path": os.path.join(BASE_PROJECT_DIR, "HTP_Datasets", "women", "test"),
        "main_obj_id": 2,
        "parts_map": {11: "eye", 12: "leg", 13: "mouth", 14: "arm"},
        "group_for_size_loc": "person",
    },
    {
        "name": "tree",
        "path": os.path.join(BASE_PROJECT_DIR, "HTP_Datasets", "tree", "test"),
        "main_obj_id": 1,
        "parts_map": {6: "branch", 7: "root", 8: "crown", 9: "fruit", 10: "gnarl"},
        "group_for_size_loc": "tree",
    },
]

# 2. 스크립트 실행 준비
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✅ 결과는 '{OUTPUT_DIR}' 폴더에 저장됩니다.")

# YOLO 모델 로드
print(f"모델 로딩 중... ({MODEL_PATH})")
try:
    model = YOLO(MODEL_PATH)
    print("✅ 모델 로딩 완료!")
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")
    exit()

# 3. 각 폴더별로 이미지 예측 및 분석
for task in TASKS:
    task_name = task["name"]
    image_dir = task["path"]
    main_obj_id = task["main_obj_id"]
    parts_map = task["parts_map"]
    group_name = task["group_for_size_loc"]

    print(f"\n--- '{task_name}' 그룹 분석 시작 ---")

    if not os.path.isdir(image_dir):
        print(f"⚠️ 경고: '{image_dir}' 폴더를 찾을 수 없습니다. 이 그룹을 건너뜁니다.")
        continue

    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"✅ 총 {len(image_files)}개의 이미지 파일을 찾았습니다.")

    results_list = []
    for i, image_file in enumerate(image_files):
        print(f"  ... [{i+1}/{len(image_files)}] '{image_file}' 분석 중 ...")
        image_path = os.path.join(image_dir, image_file)

        detections = model(image_path, verbose=False, conf=0.7, iou=0.7)

        # 결과 레코드 초기화
        record = {"id": image_file, "size": "N/A", "loc": "N/A"}
        for part_name in parts_map.values():
            if part_name == "window":
                # window_cnt의 기본값을 'absence'로 설정
                record["window_cnt"] = "absence"
            else:
                record[f"{part_name}_yn"] = "n"

        if detections and detections[0].boxes:
            img_h, img_w = detections[0].orig_shape
            detected_boxes = detections[0].boxes

            # 1. 메인 객체 분석
            main_object_candidates = [
                box for box in detected_boxes if int(box.cls) == main_obj_id
            ]
            if main_object_candidates:
                largest_box = max(
                    main_object_candidates,
                    key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0])
                    * (b.xyxy[0][3] - b.xyxy[0][1]),
                )
                pixel_box = largest_box.xyxy[0].cpu().numpy().astype(int)
                size, loc = analyze_size_and_loc(img_w, img_h, pixel_box, group_name)
                record["size"] = size
                record["loc"] = loc

            # 2. 부속 객체 처리
            #  window 개수를 세기 위한 임시 카운터 변수 추가
            window_counter = 0
            for box in detected_boxes:
                class_id = int(box.cls)
                if class_id in parts_map:
                    part_name = parts_map[class_id]
                    if part_name == "window":
                        window_counter += 1  # 실제 개수만 카운트
                    else:
                        record[f"{part_name}_yn"] = "y"

            # 루프가 끝난 후, window 개수에 따라 최종 텍스트 값을 할당
            if "window_cnt" in record:
                if window_counter >= 3:
                    record["window_cnt"] = "more than 3"
                elif window_counter > 0:  # 1 또는 2
                    record["window_cnt"] = "1 or 2"
                # window_counter가 0이면 초기값 "absence" 유지

        results_list.append(record)

    # 4. 데이터프레임 변환 및 CSV 저장
    if not results_list:
        print(
            f"'{task_name}' 그룹에서 분석할 데이터가 없어 CSV 파일을 생성하지 않습니다."
        )
        continue

    df = pd.DataFrame(results_list)

    part_columns = []
    for part_name in sorted(list(parts_map.values())):
        if part_name == "window":
            part_columns.append("window_cnt")
        else:
            part_columns.append(f"{part_name}_yn")

    column_order = ["id", "size", "loc"] + part_columns
    df = df.reindex(columns=column_order)

    csv_path = os.path.join(OUTPUT_DIR, f"{task_name}_labeling_result.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(
        f"'{task_name}' 그룹 분석 완료! 결과가 다음 파일로 저장되었습니다:\n    {csv_path}"
    )

print("\n🎉 모든 작업이 완료되었습니다!")
