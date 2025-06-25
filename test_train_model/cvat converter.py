import xml.etree.ElementTree as ET
import pandas as pd

# 1. XML 파일 파싱
tree = ET.parse("annotations.xml")
root = tree.getroot()

# 추출한 데이터를 저장할 리스트
processed_data = []

# 2. 각 이미지(image) 태그를 순회
for image_tag in root.findall("image"):
    image_name = image_tag.get("name")
    image_id = image_tag.get("id")

    # 3. 이미지 태그 안의 각 박스(box) 태그를 순회
    # 각 box가 나무 하나에 해당함
    box_id_counter = 0
    for box_tag in image_tag.findall("box"):
        # 하나의 나무 객체 정보를 담을 딕셔너리
        tree_object = {}

        # 고유 ID 생성 (하나의 이미지에 나무가 여러 개일 수 있으므로)
        unique_object_id = f"{image_id}_{box_id_counter}"
        tree_object["object_id"] = unique_object_id
        tree_object["image_name"] = image_name

        # 좌표 정보 (필요한 경우)
        tree_object["xtl"] = box_tag.get("xtl")
        tree_object["ytl"] = box_tag.get("ytl")
        tree_object["xbr"] = box_tag.get("xbr")
        tree_object["ybr"] = box_tag.get("ybr")

        # 4. 박스 태그 안의 속성(attribute)들을 추출
        for attr_tag in box_tag.findall("attribute"):
            attr_name = attr_tag.get("name")
            # CVAT에서는 'true'/'false' 문자열로 저장되므로 boolean으로 변환
            attr_value = True if attr_tag.text == "true" else False
            tree_object[attr_name] = attr_value

        processed_data.append(tree_object)
        box_id_counter += 1

# 5. 결과 확인 (Pandas DataFrame으로 변환하면 보기 좋음)
df = pd.DataFrame(processed_data)
print(df)

# CSV 파일로 저장하여 다음 단계(분류 모델 학습 데이터셋 구축)에 사용
df.to_csv("extracted_attributes.csv", index=False)
