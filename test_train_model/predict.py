# 이 스크립트는 학습된 모델을 사용하여 새로운 나무 이미지에 대해 속성을 예측합니다.
# 모델 구조는 학습 스크립트와 동일하며, 가중치만 불러옵니다.
# 예측할 이미지 경로를 지정하고, 모델 가중치를 불러온 후
# 이미지를 전처리하여 속성을 예측합니다.
# 예측 결과는 각 속성에 대해 'Yes' 또는 'No'로 출력합니다.

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os


# --- 1. 설정 (Configuration) ---
# 학습 때와 동일한 설정을 사용해야 합니다.
class CFG:
    MODEL_NAME = "mobilenet_v3_small"
    IMAGE_SIZE = 224
    NUM_ATTRIBUTES = 5  # branch, root, crown, fruit, gnarl


# 속성 이름 정의 (학습 시 사용한 순서와 동일해야 함)
ATTRIBUTE_NAMES = [
    "가지(Branch)",
    "뿌리(Root)",
    "수관(Crown)",
    "열매(Fruit)",
    "옹이(Gnarl)",
]


# --- 2. 모델 정의 (학습 스크립트와 동일한 함수) ---
def get_model(num_attributes, pretrained=False):
    """
    사전 학습된 모델을 불러와 마지막 레이어를 우리 작업에 맞게 수정
    (주의: 가중치를 불러오려면 모델의 구조가 완벽히 동일해야 함)
    """
    # MobileNetV3 Small 모델 구조만 불러오기 (가중치는 나중에 로드)
    model = models.mobilenet_v3_small(weights=None)

    # 마지막 분류 레이어(classifier)를 새로운 레이어로 교체
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_attributes)
    return model


# --- 3. 예측 함수 정의 ---
def predict_attributes(model, image_path, transform, device):
    """
    단일 이미지에 대해 속성을 예측하는 함수
    """
    try:
        # 이미지 열기 및 전처리
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image)

        # 모델은 배치(batch) 입력을 가정하므로, 차원 추가 [C, H, W] -> [B, C, H, W]
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # 예측 수행 (그래디언트 계산 비활성화)
        with torch.no_grad():
            outputs = model(image_tensor)

        # 출력(logits)을 확률(0~1)로 변환
        probabilities = torch.sigmoid(outputs)

        # 확률을 CPU로 이동하고 리스트로 변환
        probabilities = probabilities.squeeze().cpu().numpy().tolist()

        return probabilities

    except FileNotFoundError:
        print(f"오류: 이미지 파일을 찾을 수 없습니다 - {image_path}")
        return None
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {e}")
        return None


# --- 4. 메인 실행 로직 ---
if __name__ == "__main__":
    # --- 설정 ---
    MODEL_WEIGHTS_PATH = "best_mobilenet_v3_small_model.pth"

    # <<< 예측할 이미지 경로를 여기에 입력하세요 >>>
    TEST_IMAGE_PATH = "test_images/tree_7_female_12879.jpg"

    PREDICTION_THRESHOLD = 0.5  # 이 값보다 확률이 높으면 'Yes'로 판단

    # --- 준비 ---
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 모델 구조 생성 및 가중치 불러오기
    try:
        model = get_model(num_attributes=CFG.NUM_ATTRIBUTES)
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model.to(device)
        model.eval()  # <<< 매우 중요: 모델을 평가 모드(evaluation mode)로 설정
        print(f"모델 가중치를 '{MODEL_WEIGHTS_PATH}'에서 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        print(f"오류: 모델 가중치 파일 '{MODEL_WEIGHTS_PATH}'를 찾을 수 없습니다.")
        exit()  # 모델이 없으면 실행 중단

    # 이미지 변환 정의 (학습 시 검증(validation) 데이터에 사용했던 것과 동일해야 함)
    image_transform = transforms.Compose(
        [
            transforms.Resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)),
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
        for i, attr_name in enumerate(ATTRIBUTE_NAMES):
            prob = probabilities[i]
            prediction = "Yes" if prob >= PREDICTION_THRESHOLD else "No"
            print(f"{attr_name:<10s} : {prediction:<5s} (신뢰도: {prob * 100:.2f}%)")
        print("-----------------")
