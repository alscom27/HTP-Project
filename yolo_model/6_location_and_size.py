import os
import pandas as pd
import cv2


# -------------------------------------------------------------------
#                               분석 함수
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
#                               메인 스크립트
# ===================================================================
print("--- 스크립트 실행 시작 ---")

# 1. ❗ 사용자 설정 ❗
BASE_PROJECT_DIR = r"C:\Users\main\Desktop\HTP-Project"
IMAGE_DIR = os.path.join(BASE_PROJECT_DIR, "visualization", "images", "images")
LABEL_DIR = os.path.join(BASE_PROJECT_DIR, "visualization", "images", "labels")
OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, "result")

MAIN_OBJECT_IDS = {"person": 2, "tree": 1, "house": 0}
PERSON_PARTS_MAP = {11: "eye", 12: "leg", 13: "mouth", 14: "arm"}
TREE_PARTS_MAP = {6: "branch", 7: "root", 8: "crown", 9: "fruit", 10: "gnarl"}
HOUSE_PARTS_MAP = {3: "door", 4: "roof", 5: "window"}  # ❗ window의 class ID는 5입니다.

# 2. 스크립트 실행 준비
os.makedirs(OUTPUT_DIR, exist_ok=True)
results_list = []

all_labels = (
    list(PERSON_PARTS_MAP.values())
    + list(TREE_PARTS_MAP.values())
    + list(HOUSE_PARTS_MAP.values())
)

target_prefixes = ("men", "women", "tree", "house")
image_files = [
    f
    for f in os.listdir(IMAGE_DIR)
    if f.lower().startswith(target_prefixes)
    and f.lower().endswith((".png", ".jpg", ".jpeg"))
]

print(f"✅ 분석 대상 이미지 총 {len(image_files)}개를 찾았습니다.")

# 3. 라벨 파일 분석
print("\n라벨 파일 분석 (크기/위치 포함)을 시작합니다...")
for i, image_file in enumerate(image_files):
    if (i + 1) % 100 == 0:
        print(f"   ... {i+1}/{len(image_files)} 처리 중 ...")

    # ❗ 초기값 설정 변경: window는 absence, 나머지는 n으로 시작
    record = {"filename": image_file, "group": "unknown", "size": None, "loc": None}
    for label in all_labels:
        if label == "window":
            record[f"{label}_cnt"] = "absence"  # window의 기본값을 'absence'로 설정
        else:
            record[f"{label}_yn"] = "n"

    filename_lower = image_file.lower()
    current_group = None
    if filename_lower.startswith(("men", "women")):
        current_group = "person"
    elif filename_lower.startswith("tree"):
        current_group = "tree"
    elif filename_lower.startswith("house"):
        current_group = "house"
    if not current_group:
        continue
    record["group"] = current_group

    base_filename = os.path.splitext(image_file)[0]
    label_path = os.path.join(LABEL_DIR, f"{base_filename}.txt")

    if os.path.exists(label_path):
        image_path = os.path.join(IMAGE_DIR, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue
        img_h, img_w, _ = image.shape

        detected_objects = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    yolo_box = [float(p) for p in parts[1:]]
                    detected_objects.append({"id": class_id, "box": yolo_box})

        main_obj_id = MAIN_OBJECT_IDS[current_group]
        for obj in detected_objects:
            if obj["id"] == main_obj_id:
                x_c, y_c, w, h = obj["box"]
                x_min = int((x_c - w / 2) * img_w)
                y_min = int((y_c - h / 2) * img_h)
                x_max = int((x_c + w / 2) * img_w)
                y_max = int((y_c + h / 2) * img_h)
                pixel_box = [x_min, y_min, x_max, y_max]

                size, loc = analyze_size_and_loc(img_w, img_h, pixel_box, current_group)
                record["size"] = size
                record["loc"] = loc
                break

        # ❗ 그룹별 부속 객체 처리 로직 수정
        found_ids = {obj["id"] for obj in detected_objects}

        if current_group == "house":
            # ❗ Window 개수 세기 로직 추가
            window_id = 5  # window의 class ID
            window_count = sum(1 for obj in detected_objects if obj["id"] == window_id)

            if window_count >= 3:
                record["window_cnt"] = "more than 3"
            elif window_count > 0:  # 1 또는 2개일 경우
                record["window_cnt"] = "1 or 2"
            # 0개일 경우는 초기값 'absence' 유지

            # ❗ Window를 제외한 나머지 house 파트 처리
            other_house_parts_map = {
                id: name for id, name in HOUSE_PARTS_MAP.items() if id != window_id
            }
            for part_id, part_name in other_house_parts_map.items():
                if part_id in found_ids:
                    record[f"{part_name}_yn"] = "y"

        elif current_group == "person":
            for part_id, part_name in PERSON_PARTS_MAP.items():
                if part_id in found_ids:
                    record[f"{part_name}_yn"] = "y"

        elif current_group == "tree":
            for part_id, part_name in TREE_PARTS_MAP.items():
                if part_id in found_ids:
                    record[f"{part_name}_yn"] = "y"

    results_list.append(record)

print("✅ 모든 라벨 파일 분석 완료!")

# 4. 데이터프레임 변환 및 CSV 저장
df = pd.DataFrame(results_list)
feature_columns = []
for label in all_labels:
    if label == "window":
        feature_columns.append(f"{label}_cnt")
    else:
        feature_columns.append(f"{label}_yn")

column_order = ["filename", "group", "size", "loc"] + feature_columns
df = df.reindex(columns=column_order)

csv_path = os.path.join(OUTPUT_DIR, "htp_analysis_final.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"✅ 최종 분석 결과가 CSV 파일로 저장되었습니다: {csv_path}")

print("\n🎉 모든 작업이 완료되었습니다!")
