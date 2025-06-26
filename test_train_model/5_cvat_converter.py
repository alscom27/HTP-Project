# 이 코드는 지정된 폴더에 있는 모든 CVAT XML 파일을 파싱하여 객체 속성을 추출하고,
# 모든 파일의 데이터를 병합하여 하나의 CSV 파일로 저장합니다.

import xml.etree.ElementTree as ET
import pandas as pd
import os
import glob

# 1. XML 파일들이 있는 폴더 경로를 지정합니다.
# '.'는 현재 폴더를 의미합니다. 다른 폴더에 있다면 경로를 수정하세요. (예: "C:/Users/Downloads/annotations")
xml_folder_path = "C:/Users/UserK/Desktop/HTP-Project/yeeun/annotations"

# 지정된 폴더에서 .xml 확장자를 가진 모든 파일 목록을 가져옵니다.
xml_files = glob.glob(os.path.join(xml_folder_path, "*.xml"))

if not xml_files:
    print(
        f"Error: '{xml_folder_path}' 폴더에서 XML 파일을 찾을 수 없습니다. 폴더 경로를 확인해주세요."
    )
    exit()

print(f"총 {len(xml_files)}개의 XML 파일을 찾았습니다.")

# 모든 파일에서 추출한 데이터를 저장할 리스트
all_processed_data = []

# 2. 찾은 XML 파일들을 하나씩 순회
for xml_file in xml_files:
    print(f"--- 처리 중인 파일: {xml_file} ---")
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError:
        print(f"Warning: '{xml_file}' 파일을 파싱하는 데 실패했습니다. 건너뜁니다.")
        continue

    file_basename = os.path.basename(xml_file)  # 고유 ID 생성을 위해 파일명 추출

    # 3. 각 이미지(image) 태그를 순회
    for image_tag in root.findall("image"):
        image_name = image_tag.get("name")
        image_id = image_tag.get("id")

        # 4. 이미지 태그 안의 각 박스(box) 태그를 순회
        # 각 box가 하나의 객체(나무, 집, 사람 등)에 해당함
        box_id_counter = 0
        for box_tag in image_tag.findall("box"):
            # 하나의 객체 정보를 담을 딕셔너리
            object_data = {}

            # 라벨(객체 종류) 정보 추출
            label = box_tag.get("label")

            # 고유 ID 생성 (파일명 + 이미지 ID + 박스 카운터)
            unique_object_id = f"{file_basename}_{image_id}_{box_id_counter}"
            object_data["object_id"] = unique_object_id
            object_data["source_file"] = file_basename  # 출처 파일명 추가
            object_data["image_name"] = image_name
            object_data["label"] = label

            # 좌표 정보 (필요한 경우)
            object_data["xtl"] = box_tag.get("xtl")
            object_data["ytl"] = box_tag.get("ytl")
            object_data["xbr"] = box_tag.get("xbr")
            object_data["ybr"] = box_tag.get("ybr")

            # 5. 박스 태그 안의 속성(attribute)들을 먼저 모두 추출하여 딕셔너리로 저장
            attributes = {
                attr.get("name"): attr.text for attr in box_tag.findall("attribute")
            }

            # 6. 객체 라벨에 따라 필요한 속성을 선택적으로 처리
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

            all_processed_data.append(object_data)
            box_id_counter += 1

# 7. 병합된 결과 확인 (Pandas DataFrame으로 변환)
# 각 객체 타입별로 속성 수가 다르므로, 해당 없는 속성은 NaN으로 표시됨
if not all_processed_data:
    print("처리할 데이터가 없습니다. XML 파일의 내용을 확인해주세요.")
    exit()

df = pd.DataFrame(all_processed_data)
print("\n---------- 병합 및 추출된 최종 데이터프레임 ----------")
print(df)

# 8. CSV 파일로 저장
# 파일 이름을 'merged_attributes.csv'로 고정
output_filename = "merged_attributes.csv"
df.to_csv(output_filename, index=False, encoding="utf-8-sig")

print(f"\n성공적으로 모든 데이터를 병합하여 '{output_filename}' 파일로 저장했습니다.")
