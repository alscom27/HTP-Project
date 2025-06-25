# 주어진 CSV 파일과 원본 이미지 폴더를 기반으로 이미지를 자르고
# 나무 객체의 속성 분류 학습용 데이터셋을 생성하는 스크립트입니다.
# 이미지의 bbox 좌표 정보를 사용하여 이미지를 자르고,
# 각 이미지에 대한 다중 레이블(속성)을 생성하여 CSV 파일로 저장합니다.

import pandas as pd
from PIL import Image
import os
from tqdm import tqdm  # 진행 상황을 예쁘게 보여주는 라이브러리 (pip install tqdm)


def create_classifier_dataset(csv_path, images_dir, output_dir):
    """
    CSV 파일과 원본 이미지를 기반으로, 이미지를 자르고
    다중 레이블 분류 학습용 데이터셋을 생성합니다.

    Args:
        csv_path (str): 속성 정보가 담긴 CSV 파일 경로.
        images_dir (str): 원본 이미지가 있는 폴더 경로.
        output_dir (str): 결과물이 저장될 폴더 경로.
    """
    # 1. 경로 설정 및 출력 폴더 생성
    cropped_images_output_dir = os.path.join(output_dir, "cropped_images")
    output_csv_path = os.path.join(output_dir, "classifier_labels.csv")

    if not os.path.exists(cropped_images_output_dir):
        os.makedirs(cropped_images_output_dir)
        print(f"'{cropped_images_output_dir}' 폴더를 생성했습니다.")

    # 2. CSV 파일 읽기
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"오류: '{csv_path}' 파일을 찾을 수 없습니다.")
        return

    # 3. 최종 라벨 데이터를 저장할 리스트
    classifier_data = []

    # 속성 컬럼 이름들 정의 (순서가 중요!)
    attribute_columns = ["branch_yn", "root_yn", "crown_yn", "fruit_yn", "gnarl_yn"]

    print("이미지 자르기 및 라벨 생성을 시작합니다...")

    # 4. CSV의 각 행을 순회하며 작업 수행 (tqdm으로 진행상황 표시)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        image_name = row["image_name"]
        source_image_path = os.path.join(images_dir, image_name)

        # 원본 이미지 파일 존재 여부 확인
        if not os.path.exists(source_image_path):
            print(f"경고: 원본 이미지 '{source_image_path}'를 찾을 수 없어 건너뜁니다.")
            continue

        try:
            # 5. 이미지 열기 및 좌표 변환
            with Image.open(source_image_path) as img:
                # 좌표를 float으로, 그리고 int로 변환
                left = int(float(row["xtl"]))
                top = int(float(row["ytl"]))
                right = int(float(row["xbr"]))
                bottom = int(float(row["ybr"]))

                # 6. 이미지 자르기 (Crop)
                cropped_img = img.crop((left, top, right, bottom))

                # 7. 잘라낸 이미지 저장
                # 파일명은 중복되지 않도록 고유 ID(object_id) 사용
                cropped_filename = f"{row['object_id']}.png"
                cropped_output_path = os.path.join(
                    cropped_images_output_dir, cropped_filename
                )
                cropped_img.save(cropped_output_path, "PNG")

                # 8. 다중-핫 인코딩 라벨 생성
                # [True, True, False, ...] -> [1, 1, 0, ...]
                label_vector = [int(row[col]) for col in attribute_columns]

                # 9. 최종 데이터 리스트에 추가
                classifier_data.append(
                    {
                        "cropped_image_path": os.path.join(
                            "cropped_images", cropped_filename
                        ),
                        "labels": label_vector,
                    }
                )

        except Exception as e:
            print(f"오류: '{image_name}' 처리 중 문제 발생 - {e}")

    # 10. 최종 라벨 CSV 파일 생성
    if classifier_data:
        classifier_df = pd.DataFrame(classifier_data)
        classifier_df.to_csv(output_csv_path, index=False)
        print("\n작업 완료!")
        print(
            f"총 {len(classifier_df)}개의 잘라낸 이미지가 '{cropped_images_output_dir}'에 저장되었습니다."
        )
        print(f"학습용 라벨 파일이 '{output_csv_path}'에 생성되었습니다.")
    else:
        print("\n처리할 데이터가 없습니다.")


if __name__ == "__main__":
    # --- 설정 값 ---
    INPUT_CSV_PATH = "extracted_attributes.csv"
    ORIGINAL_IMAGES_DIR = "original_images"
    OUTPUT_DATASET_DIR = "classifier_dataset"

    # 함수 실행
    create_classifier_dataset(INPUT_CSV_PATH, ORIGINAL_IMAGES_DIR, OUTPUT_DATASET_DIR)
