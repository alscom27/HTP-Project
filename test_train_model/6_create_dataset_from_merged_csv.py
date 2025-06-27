# 객체 라벨(house, person, tree)에 따른 csv 파일과 원본 이미지 폴더를 기반으로
# 이미지를 자르고 객체 속성 분류 학습용 데이터셋을 생성하는 스크립트입니다.

import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import glob


def merge_attribute_csvs(output_path):
    """
    현재 폴더에서 'house_attributes.csv', 'person_attributes.csv', 'tree_attributes.csv'
    파일들을 찾아 병합하고, 지정된 경로에 저장합니다.

    Args:
        output_path (str): 병합된 CSV 파일을 저장할 경로

    Returns:
        bool: 성공적으로 병합되면 True, 아니면 False
    """
    # 특정 파일들을 찾습니다.
    target_files = [
        "house_attributes.csv",
        "person_attributes.csv",
        "tree_attributes.csv",
    ]
    found_files = [f for f in target_files if os.path.exists(f)]

    if not found_files:
        print("오류: 병합할 속성 CSV 파일(house, person, tree)을 찾을 수 없습니다.")
        print(
            "현재 폴더에 'house_attributes.csv', 'person_attributes.csv', 'tree_attributes.csv' 파일이 있는지 확인해주세요."
        )
        return False

    print(f"다음 파일들을 병합합니다: {found_files}")

    try:
        df_list = [pd.read_csv(f) for f in found_files]
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.to_csv(output_path, index=False)
        print(f"성공적으로 '{output_path}' 파일에 병합된 데이터를 저장했습니다.")
        return True
    except Exception as e:
        print(f"CSV 파일 병합 중 오류 발생: {e}")
        return False


def create_classifier_dataset(csv_path, images_dir, output_dir):
    """
    하나의 병합된 CSV 파일과 원본 이미지를 기반으로, 이미지를 자르고
    분류 학습용 데이터셋을 생성합니다.

    Args:
        csv_path (str): 병합된 속성 정보가 담긴 CSV 파일 경로 ('merged_attributes.csv')
        images_dir (str): 원본 이미지가 있는 폴더 경로
        output_dir (str): 결과물이 저장될 폴더 경로
    """
    # 1. 경로 설정
    base_cropped_dir = os.path.join(output_dir, "cropped_images")
    output_csv_path = os.path.join(output_dir, "classifier_labels.csv")

    if not os.path.exists(base_cropped_dir):
        os.makedirs(base_cropped_dir)
        print(f"'{base_cropped_dir}' 폴더를 생성했습니다.")

    # 2. 병합된 CSV 파일 하나를 읽기
    try:
        df = pd.read_csv(csv_path)
        print(f"성공적으로 '{csv_path}' 파일을 읽었습니다.")
    except FileNotFoundError:
        print(f"오류: 입력 CSV 파일 '{csv_path}'를 찾을 수 없습니다.")
        return

    # 3. 최종 라벨 데이터를 저장할 딕셔너리 리스트
    classifier_data = []

    print("이미지 자르기 및 라벨 생성을 시작합니다...")

    # 4. DataFrame의 각 행을 순회하며 작업 수행
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        image_name = row["image_name"]
        source_image_path = os.path.join(images_dir, image_name)
        label = row["label"]

        if not os.path.exists(source_image_path):
            print(f"경고: 원본 이미지 '{source_image_path}'를 찾을 수 없어 건너뜁니다.")
            continue

        specific_cropped_dir = os.path.join(base_cropped_dir, label)
        if not os.path.exists(specific_cropped_dir):
            os.makedirs(specific_cropped_dir)

        try:
            # 5. 이미지 열기 및 좌표 변환
            with Image.open(source_image_path) as img:
                left = int(float(row["xtl"]))
                top = int(float(row["ytl"]))
                right = int(float(row["xbr"]))
                bottom = int(float(row["ybr"]))

                # 6. 이미지 자르기 (Crop) 및 저장
                cropped_img = img.crop((left, top, right, bottom))
                # object_id는 이미 파일명까지 포함하여 고유함
                cropped_filename = f"{row['object_id']}.png"
                cropped_output_path = os.path.join(
                    specific_cropped_dir, cropped_filename
                )
                cropped_img.save(cropped_output_path, "PNG")

                # 7. 라벨링을 위한 데이터 준비
                label_info = {
                    "image_path": os.path.relpath(
                        cropped_output_path, output_dir
                    ).replace(os.sep, "/"),
                    "label": label,
                }

                # 8. 객체 라벨에 따라 다르게 라벨링 수행
                if label == "tree":
                    # '..._yn'으로 끝나고 값이 비어있지 않은 모든 속성 컬럼을 처리
                    attr_cols = [
                        col
                        for col in row.index
                        if col.endswith("_yn") and pd.notna(row[col])
                    ]
                    for col in attr_cols:
                        label_info[col] = int(row[col])

                elif label == "house":
                    bool_cols = ["door_yn", "roof_yn"]
                    for col in bool_cols:
                        if col in row and pd.notna(row[col]):
                            label_info[col] = int(row[col])

                    # 'window_cnt'를 원-핫 인코딩으로 변환
                    if "window_cnt" in row and pd.notna(row["window_cnt"]):
                        window_categories = ["absence", "1 or 2", "more than 3"]
                        for cat in window_categories:
                            col_name = f"window_cnt_{cat.replace(' ', '_')}"
                            label_info[col_name] = (
                                1 if str(row["window_cnt"]) == cat else 0
                            )

                elif label in ["men", "women", "person"]:
                    attr_cols = [
                        "eye_yn",
                        "leg_yn",
                        "mouth_yn",
                        "arm_yn",
                    ]
                    for col in attr_cols:
                        if col in row and pd.notna(row[col]):
                            label_info[col] = int(row[col])

                classifier_data.append(label_info)

        except Exception as e:
            print(
                f"오류: '{image_name}' (ID: {row['object_id']}) 처리 중 문제 발생 - {e}"
            )

    # 9. 최종 라벨 CSV 파일 생성
    if classifier_data:
        classifier_df = pd.DataFrame(classifier_data)
        # image_path, label 컬럼을 제외한 모든 컬럼의 빈 값을 0으로 채우고 정수형으로 변환
        label_columns = classifier_df.columns.drop(["image_path", "label"])
        classifier_df[label_columns] = (
            classifier_df[label_columns].fillna(0).astype(int)
        )

        # 컬럼 순서 재정렬 (image_path, label을 맨 앞으로)
        first_cols = ["image_path", "label"]
        other_cols = sorted(
            [col for col in classifier_df.columns if col not in first_cols]
        )
        classifier_df = classifier_df[first_cols + other_cols]

        classifier_df.to_csv(output_csv_path, index=False)
        print("작업 완료!")
        print(
            f"총 {len(classifier_df)}개의 잘라낸 이미지가 '{base_cropped_dir}' 내 하위 폴더에 저장되었습니다."
        )
        print(f"통합 학습용 라벨 파일이 '{output_csv_path}'에 생성되었습니다.")
    else:
        print("처리할 데이터가 없습니다.")


if __name__ == "__main__":
    # --- 설정 값 ---
    MERGED_CSV_PATH = "merged_attributes.csv"

    # 1. 지정된 속성 CSV 파일들 병합
    if not merge_attribute_csvs(MERGED_CSV_PATH):
        # 병합할 파일이 없으면 스크립트 중단
        exit()

    # 원본 이미지가 저장된 폴더
    ORIGINAL_IMAGES_DIR = r"C:\Users\UserK\Desktop\HTP-Project\yeeun\original_images"

    # 최종 데이터셋(잘린 이미지, 라벨 파일)이 저장될 폴더
    OUTPUT_DATASET_DIR = "classifier_dataset"

    # 함수 실행
    create_classifier_dataset(MERGED_CSV_PATH, ORIGINAL_IMAGES_DIR, OUTPUT_DATASET_DIR)
