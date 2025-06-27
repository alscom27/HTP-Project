# 이 스크립트는 학습된 통합 모델을 사용하여 새로운 이미지에 대해 모든 속성을 예측합니다.

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os


# --- 1. 모델 정의 (학습 스크립트와 동일한 함수) ---
def get_model(model_name, num_attributes, pretrained=False):
    """
    사전 학습된 모델 구조를 불러와 마지막 레이어를 우리 작업에 맞게 수정
    """
    model = None
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=None
        )  # 가중치는 파일에서 불러오므로 None
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_attributes)
    # 다른 모델 아키텍처를 사용하는 경우 여기에 추가
    # elif model_name == "mobilenet_v3_small":
    #     ...
    else:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {model_name}")
    return model


# --- 2. 예측 함수 정의 ---
def predict_attributes(model, image_path, transform, device):
    """
    단일 이미지에 대해 속성을 예측하는 함수
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)

        # 시그모이드 함수를 적용하여 확률값으로 변환
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy().tolist()

        # 속성이 하나일 경우, tolist()가 float을 반환하므로 리스트로 감싸줌
        if not isinstance(probabilities, list):
            probabilities = [probabilities]

        return probabilities

    except FileNotFoundError:
        print(f"오류: 이미지 파일을 찾을 수 없습니다 - {image_path}")
        return None
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {e}")
        return None


# --- 3. 메인 실행 로직 ---
if __name__ == "__main__":
    # ===================================================================
    #                           사용자 설정
    # ===================================================================

    # <<< 1. 예측할 이미지 경로를 여기에 입력하세요 >>>
    # 예: "test_images/tree_1.jpg", "path/to/your/house_image.png"
    TEST_IMAGE_PATH = "test_images/tree_7_male_01628.jpg"  # 예측할 이미지 경로로 수정

    # --- (고급 설정) ---
    # 학습 시 사용한 모델 이름
    BASE_MODEL_NAME = "efficientnet_b0"

    # 학습된 통합 모델 가중치 파일 경로
    MODEL_WEIGHTS_PATH = f"best_unified_model_{BASE_MODEL_NAME}.pth"

    # 학습 시 사용한 이미지 크기
    IMAGE_SIZE = 224
    # 예측 임계값 (이 값보다 확률이 높으면 'Yes'로 판단)
    PREDICTION_THRESHOLD = 0.5

    # ===================================================================

    # --- 준비 단계 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    print(f"사용 모델 파일: '{MODEL_WEIGHTS_PATH}'")

    # 모델 가중치 및 속성 정보 불러오기
    try:
        checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
        attribute_columns = checkpoint["attribute_columns"]
        num_attributes = len(attribute_columns)

        model = get_model(model_name=BASE_MODEL_NAME, num_attributes=num_attributes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()  # <<< 매우 중요: 모델을 평가 모드로 설정

        print(f"모델 가중치를 성공적으로 불러왔습니다.")
        print(f"모델이 예측하는 속성 ({num_attributes}개): {attribute_columns}")

    except FileNotFoundError:
        print(f"오류: 모델 가중치 파일 '{MODEL_WEIGHTS_PATH}'를 찾을 수 없습니다.")
        print("7번 스크립트로 학습을 먼저 실행하여 모델 파일을 생성했는지 확인하세요.")
        exit()
    except KeyError:
        print(f"오류: 모델 파일 '{MODEL_WEIGHTS_PATH}'이 올바른 형식이 아닙니다.")
        print(
            "'model_state_dict'와 'attribute_columns' 키가 포함되어 있는지 확인하세요."
        )
        exit()

    # 이미지 변환 정의 (학습 시 검증 데이터에 사용했던 것과 동일)
    image_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # --- 예측 실행 ---
    print(f"\n이미지 예측 중: '{TEST_IMAGE_PATH}'")
    probabilities = predict_attributes(model, TEST_IMAGE_PATH, image_transform, device)

    # --- 결과 출력 ---
    if probabilities:
        print("\n--- 예측 결과 ---")
        for i, attr_name in enumerate(attribute_columns):
            prob = probabilities[i]
            prediction = "Yes" if prob >= PREDICTION_THRESHOLD else "No"
            # 속성 이름에서 '_'를 공백으로, 'yn'을 제거하여 더 읽기 쉽게 만듭니다.
            display_name = attr_name.replace("_", " ").replace(" yn", "").capitalize()
            print(f"{display_name:<25s} : {prediction:<5s} (신뢰도: {prob * 100:.2f}%)")
        print("-----------------")
