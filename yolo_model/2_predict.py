import os
import glob
from ultralytics import YOLO

# --- 설정 부분 ---

# 1. 불러올 모델 경로
MODEL_PATH = "best.pt"

# 2. 이미지가 저장된 폴더 경로
IMAGE_DIR = "test"

# 3. 예측할 이미지 파일명의 시작 부분 (클래스별 접두사)
# 'men'과 'women'은 'person' 클래스에 해당할 것으로 보입니다.
PREFIXES = ["house", "tree", "men", "women"]

# 4. 각 접두사별로 선택할 이미지 개수
IMAGES_PER_PREFIX = 100

# --- 코드 실행 부분 ---

# YOLO 모델 불러오기
# 모델 파일이 없으면 오류가 발생하므로 경로를 정확히 확인해주세요.
try:
    model = YOLO(MODEL_PATH)
    # 모델의 클래스 이름 가져오기 (결과 출력 시 사용)
    class_names = model.names
    print(f"모델 로딩 완료: {MODEL_PATH}")
    print(f"클래스 이름: {class_names}")
except Exception as e:
    print(f"모델 로딩 중 오류 발생: {e}")
    print("MODEL_PATH 변수에 지정된 경로가 올바른지 확인해주세요.")
    exit()

# 예측할 이미지 파일 목록 만들기
image_paths_to_predict = []
print("\n--- 이미지 파일 선택 시작 ---")

# 이미지 폴더가 존재하는지 확인
if not os.path.isdir(IMAGE_DIR):
    print(f"오류: '{IMAGE_DIR}' 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

for prefix in PREFIXES:
    # 각 접두사로 시작하는 모든 jpg 파일 검색
    # os.path.join을 사용하여 OS에 맞는 경로 구분자로 합쳐줍니다.
    search_pattern = os.path.join(IMAGE_DIR, f"{prefix}*.jpg")
    found_files = glob.glob(search_pattern)

    # 찾은 파일 중에서 지정된 개수만큼 선택
    selected_files = found_files[:IMAGES_PER_PREFIX]
    image_paths_to_predict.extend(selected_files)

    print(
        f"'{prefix}' 시작 파일: {len(found_files)}개 발견, {len(selected_files)}개 선택"
    )

print(f"\n총 {len(image_paths_to_predict)}개의 이미지에 대해 예측을 수행합니다.")
# 예측 실행
# for img in image_paths_to_predict:
# if "house" in img:
if image_paths_to_predict:
    try:
        pred = model.predict(
            source=image_paths_to_predict,  # 이미지 파일 목록을 직접 전달
            save=True,  # 예측 결과 이미지 저장
            conf=0.5,
        )

        # 예측 결과 출력
        print("\n--- 예측 결과 ---")
        for result in pred:
            # result.path에서 파일명만 추출
            image_name = os.path.basename(result.path)
            print(f"Image: {image_name}")

            for box in result.boxes:
                # 클래스 ID를 정수로 변환
                class_id = int(box.cls.item())
                # 클래스 이름 찾기
                class_name = class_names[class_id]
                # 신뢰도와 바운딩 박스 좌표
                confidence = box.conf.item()
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                print(f"  - 탐지된 객체: '{class_name}' (Class ID: {class_id})")
                print(f"    Confidence: {confidence:.2f}")
                # 바운딩 박스 좌표를 정수로 변환하여 보기 쉽게 출력
                print(f"    BBox (xyxy): {[int(coord) for coord in bbox]}")

    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
else:
    print(
        "\n예측할 이미지를 찾지 못했습니다. IMAGE_DIR 또는 PREFIXES 설정을 확인해주세요."
    )
