# 여러 개의 CSV 파일과 원본 이미지 폴더를 기반으로 이미지를 자르고
# 다양한 객체(나무, 집, 사람)의 속성 분류 학습용 데이터셋을 생성하는 스크립트입니다.
# 지정된 폴더 내의 'extracted_*_attributes.csv' 패턴의 모든 파일을 읽어 처리합니다.

import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import glob  # 파일 경로를 쉽게 다루기 위한 라이브러리


def create_classifier_dataset(csv_dir, images_dir, output_dir):
    """
    다양한 객체 정보가 담긴 여러 CSV 파일들과 원본 이미지를 기반으로, 이미지를 자르고
    분류 학습용 데이터셋을 생성합니다.

    Args:
        csv_dir (str): 'extracted_*_attributes.csv' 파일들이 있는 폴더 경로
        images_dir (str): 원본 이미지가 있는 폴더 경로
        output_dir (str): 결과물이 저장될 폴더 경로
    """
    # 1. 경로 설정
    base_cropped_dir = os.path.join(output_dir, "cropped_images")
    output_csv_path = os.path.join(output_dir, "classifier_labels.csv")

    if not os.path.exists(base_cropped_dir):
        os.makedirs(base_cropped_dir)
        print(f"'{base_cropped_dir}' 폴더를 생성했습니다.")

    # 2. 지정된 폴더에서 CSV 파일들 찾아 하나로 합치기
    search_pattern = os.path.join(csv_dir, "extracted_*_attributes.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(
            f"오류: '{csv_dir}' 폴더에서 'extracted_*_attributes.csv' 패턴의 파일을 찾을 수 없습니다."
        )
        return

    print(f"{len(csv_files)}개의 CSV 파일을 발견했습니다: {csv_files}")

    df_list = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)

    # 3. 최종 라벨 데이터를 저장할 딕셔너리 리스트
    classifier_data = []

    print("이미지 자르기 및 라벨 생성을 시작합니다...")

    # 4. 합쳐진 DataFrame의 각 행을 순회하며 작업 수행
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
                        label_info[col] = int(row[col]) if pd.notna(row[col]) else 0

                    window_categories = ["1", "2", "more than 3"]
                    for cat in window_categories:
                        col_name = f"window_cnt_{cat.replace(' ', '_')}"
                        label_info[col_name] = (
                            1 if str(row.get("window_cnt")) == cat else 0
                        )

                elif label in ["men", "women", "person"]:
                    attr_cols = ["eye_yn", "leg_yn", "mouth_yn"]
                    for col in attr_cols:
                        label_info[col] = int(row[col]) if pd.notna(row[col]) else 0

                classifier_data.append(label_info)

        except Exception as e:
            print(
                f"오류: '{image_name}' (ID: {row['object_id']}) 처리 중 문제 발생 - {e}"
            )

    # 10. 최종 라벨 CSV 파일 생성
    if classifier_data:
        classifier_df = pd.DataFrame(classifier_data)
        label_columns = classifier_df.columns.drop(["image_path", "label"])
        classifier_df[label_columns] = (
            classifier_df[label_columns].fillna(0).astype(int)
        )

        first_cols = ["image_path", "label"]
        other_cols = sorted(
            [col for col in classifier_df.columns if col not in first_cols]
        )
        classifier_df = classifier_df[first_cols + other_cols]

        classifier_df.to_csv(output_csv_path, index=False)
        print("\n작업 완료!")
        print(
            f"총 {len(classifier_df)}개의 잘라낸 이미지가 '{base_cropped_dir}' 내 하위 폴더에 저장되었습니다."
        )
        print(f"통합 학습용 라벨 파일이 '{output_csv_path}'에 생성되었습니다.")
    else:
        print("\n처리할 데이터가 없습니다.")


if __name__ == "__main__":
    # --- 설정 값 ---
    # 이전 스크립트가 'extracted_*_attributes.csv' 파일들을 저장한 폴더
    INPUT_CSV_DIR = (
        "."  # 현재 폴더에 CSV가 있다고 가정, 필요시 "attribute_csvs" 등으로 변경
    )
    # 원본 이미지가 저장된 폴더
    ORIGINAL_IMAGES_DIR = "original_images"
    # 최종 데이터셋이 저장될 폴더
    OUTPUT_DATASET_DIR = "classifier_dataset"

    # 함수 실행
    create_classifier_dataset(INPUT_CSV_DIR, ORIGINAL_IMAGES_DIR, OUTPUT_DATASET_DIR)
