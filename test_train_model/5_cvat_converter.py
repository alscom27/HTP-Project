# 이 코드는 CVAT에서 추출한 XML 파일을 파싱하여 다양한 객체(나무, 집, 사람)의 속성을 추출하고,
# 각 객체에 대한 고유 ID를 생성합니다.
# 추출된 속성은 Pandas DataFrame으로 변환되어 하나의 CSV 파일로 저장됩니다.

import xml.etree.ElementTree as ET
import pandas as pd

# 1. XML 파일 파싱
# 실제 파일 경로에 맞게 "annotations.xml"을 수정하세요.
try:
    tree = ET.parse("annotations.xml")
    root = tree.getroot()
except FileNotFoundError:
    print("Error: 'annotations.xml' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    exit()


# 추출한 데이터를 저장할 리스트
processed_data = []

# 2. 각 이미지(image) 태그를 순회
for image_tag in root.findall("image"):
    image_name = image_tag.get("name")
    image_id = image_tag.get("id")

    # 3. 이미지 태그 안의 각 박스(box) 태그를 순회
    # 각 box가 하나의 객체(나무, 집, 사람 등)에 해당함
    box_id_counter = 0
    for box_tag in image_tag.findall("box"):
        # 하나의 객체 정보를 담을 딕셔너리
        object_data = {}

        # 라벨(객체 종류) 정보 추출
        label = box_tag.get("label")

        # 고유 ID 생성
        unique_object_id = f"{image_id}_{box_id_counter}"
        object_data["object_id"] = unique_object_id
        object_data["image_name"] = image_name
        object_data["label"] = label

        # 좌표 정보 (필요한 경우)
        object_data["xtl"] = box_tag.get("xtl")
        object_data["ytl"] = box_tag.get("ytl")
        object_data["xbr"] = box_tag.get("xbr")
        object_data["ybr"] = box_tag.get("ybr")

        # 4. 박스 태그 안의 속성(attribute)들을 먼저 모두 추출하여 딕셔너리로 저장
        attributes = {
            attr.get("name"): attr.text for attr in box_tag.findall("attribute")
        }

        # 5. 객체 라벨에 따라 필요한 속성을 선택적으로 처리
        if label == "house":
            # 집(house)일 경우: door_yn, roof_yn, window_cnt 속성 추출
            object_data["door_yn"] = (
                True if attributes.get("door_yn") == "true" else False
            )
            object_data["roof_yn"] = (
                True if attributes.get("roof_yn") == "true" else False
            )
            object_data["window_cnt"] = attributes.get(
                "window_cnt"
            )  # 텍스트 값 그대로 저장

        elif label in ["men", "women", "person"]:
            # 사람(men, women, person)일 경우: 모든 속성을 boolean으로 변환하여 저장
            for attr_name, attr_value in attributes.items():
                object_data[attr_name] = True if attr_value == "true" else False

        elif label == "tree":
            # 나무(tree)일 경우: 모든 속성을 boolean으로 변환하여 저장
            for attr_name, attr_value in attributes.items():
                object_data[attr_name] = True if attr_value == "true" else False

        # 다른 라벨이 있을 경우 여기에 elif 구문 추가 가능

        processed_data.append(object_data)
        box_id_counter += 1

# 6. 결과 확인 (Pandas DataFrame으로 변환)
# 각 객체 타입별로 속성 수가 다르므로, 해당 없는 속성은 NaN으로 표시됨
df = pd.DataFrame(processed_data)
print("---------- 추출된 데이터프레임 ----------")
print(df)

# 7. CSV 파일로 저장
# 파일 이름을 'extracted_attributes.csv'로 지정
output_filename = f"extracted_{label}_attributes.csv"
df.to_csv(output_filename, index=False, encoding="utf-8-sig")

print(f"\n성공적으로 데이터를 추출하여 '{output_filename}' 파일로 저장했습니다.")
