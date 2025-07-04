# 이 스크립트는 지정된 폴더의 모든 CVAT XML 파일을 파싱하여 객체 속성을 추출하고,
# 객체 라벨(house, person, tree)에 따라 별도의 CSV 파일로 저장합니다.

import xml.etree.ElementTree as ET
import pandas as pd
import os
import glob

# 1. XML 파일들이 있는 폴더 경로를 지정합니다.
xml_folder_path = "C:/Users/UserK/Desktop/HTP-Project/yeeun/annotations"

xml_files = glob.glob(os.path.join(xml_folder_path, "*.xml"))

if not xml_files:
    print(
        f"오류: '{xml_folder_path}' 폴더에서 XML 파일을 찾을 수 없습니다. 폴더 경로를 확인해주세요."
    )
    exit()

print(f"총 {len(xml_files)}개의 XML 파일을 찾았습니다.")

# 객체 라벨별로 데이터를 저장할 리스트
house_data = []
person_data = []
tree_data = []

# 2. 찾은 XML 파일들을 하나씩 순회
for xml_file in xml_files:
    print(f"--- 처리 중인 파일: {xml_file} ---")
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError:
        print(f"경고: '{xml_file}' 파일을 파싱하는 데 실패했습니다. 건너뜁니다.")
        continue

    file_basename = os.path.basename(xml_file)

    # 3. 각 이미지(image) 태그를 순회
    for image_tag in root.findall("image"):
        image_name = image_tag.get("name")
        image_id = image_tag.get("id")

        # 4. 이미지 태그 안의 각 박스(box) 태그를 순회
        box_id_counter = 0
        for box_tag in image_tag.findall("box"):
            object_data = {}
            label = box_tag.get("label")

            unique_object_id = (
                f"{file_basename.replace('.xml', '')}_{image_id}_{box_id_counter}"
            )
            object_data["object_id"] = unique_object_id
            object_data["image_name"] = image_name
            object_data["label"] = label
            object_data["xtl"] = box_tag.get("xtl")
            object_data["ytl"] = box_tag.get("ytl")
            object_data["xbr"] = box_tag.get("xbr")
            object_data["ybr"] = box_tag.get("ybr")

            attributes = {
                attr.get("name"): attr.text for attr in box_tag.findall("attribute")
            }

            # 5. 객체 라벨에 따라 데이터를 분리하고 속성을 1/0으로 변환
            if label == "house":
                object_data["door_yn"] = 1 if attributes.get("door_yn") == "true" else 0
                object_data["roof_yn"] = 1 if attributes.get("roof_yn") == "true" else 0
                object_data["window_cnt"] = attributes.get("window_cnt")
                house_data.append(object_data)

            elif label in ["men", "women", "person"]:
                # person으로 라벨 통일
                object_data["label"] = "person"
                for attr_name, attr_value in attributes.items():
                    object_data[attr_name] = 1 if attr_value == "true" else 0
                person_data.append(object_data)

            elif label == "tree":
                for attr_name, attr_value in attributes.items():
                    object_data[attr_name] = 1 if attr_value == "true" else 0
                tree_data.append(object_data)

            box_id_counter += 1


# 6. 각 리스트를 데이터프레임으로 변환하고 CSV 파일로 저장
def save_to_csv(data_list, filename):
    if data_list:
        df = pd.DataFrame(data_list)
        # 모든 boolean/int 타입 속성 컬럼의 NaN 값을 0으로 채움
        for col in df.columns:
            if df[col].dtype == "float":  # 보통 NaN이 있으면 float으로 인식됨
                df[col] = df[col].fillna(0).astype(int)
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"성공적으로 '{filename}' 파일을 저장했습니다. (총 {len(df)}개 객체)")
    else:
        print(f"{filename}에 해당하는 데이터를 찾지 못했습니다.")


print("\n--- 최종 파일 저장 ---")
save_to_csv(house_data, "house_attributes.csv")
save_to_csv(person_data, "person_attributes.csv")
save_to_csv(tree_data, "tree_attributes.csv")

print("\n모든 작업이 완료되었습니다.")
