# 이 스크립트는 학습된 모델을 사용하여 새로운 이미지에 대해 속성을 예측합니다.
# 사용자가 예측하려는 객체 종류(house, tree, person)를 선택하면
# 해당하는 모델과 속성 목록을 동적으로 불러와 예측을 수행합니다.

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
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_attributes)

    # <<< [수정] EfficientNet 모델 추가 >>>
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        # EfficientNet의 분류기는 model.classifier[-1]에 위치합니다.
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_attributes)

    # 다른 모델 추가 시 여기에 elif 구문 사용 (예: resnet18)
    # elif model_name == "resnet18":
    #     model = models.resnet18(weights=None)
    #     in_features = model.fc.in_features
    #     model.fc = nn.Linear(in_features, num_attributes)
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

    # <<< 1. 예측하려는 객체의 종류를 입력하세요 ('tree', 'house', 'person') >>>
    TARGET_LABEL = "tree"

    # <<< 2. 예측할 이미지 경로를 여기에 입력하세요 >>>
    # 예: "test_images/my_tree.jpg", "test_images/my_house.png"
    TEST_IMAGE_PATH = "test_images/tree_7_male_01628.jpg"  # 예측할 이미지로 경로 수정

    # --- (고급 설정) ---
    # <<< [수정] 학습 시 사용한 모델 이름을 efficientnet_b0로 변경 >>>
    BASE_MODEL_NAME = "efficientnet_b0"

    # 학습 시 사용한 이미지 크기
    IMAGE_SIZE = 224
    # 예측 임계값 (이 값보다 확률이 높으면 'Yes'로 판단)
    PREDICTION_THRESHOLD = 0.5

    # ===================================================================

    # --- 설정에 따른 변수 자동 구성 ---
    num_attributes = 0
    attribute_names = []

    if TARGET_LABEL == "tree":
        num_attributes = 5
        attribute_names = [
            "가지(Branch)",
            "뿌리(Root)",
            "수관(Crown)",
            "열매(Fruit)",
            "옹이(Gnarl)",
        ]
    elif TARGET_LABEL == "house":
        num_attributes = 5
        attribute_names = [
            "문(Door)",
            "지붕(Roof)",
            "창문 1개",
            "창문 2개",
            "창문 3개 이상",
        ]
    elif TARGET_LABEL == "person":
        num_attributes = 3
        attribute_names = ["눈(Eye)", "다리(Leg)", "입(Mouth)"]
    else:
        print(
            f"오류: 지원하지 않는 TARGET_LABEL입니다. 'tree', 'house', 'person' 중에서 선택하세요."
        )
        exit()

    # 모델 가중치 파일 경로 자동 생성
    model_weights_path = f"best_{TARGET_LABEL}_{BASE_MODEL_NAME}.pth"

    # --- 준비 단계 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    print(f"예측 대상: '{TARGET_LABEL}' | 모델 파일: '{model_weights_path}'")

    # 모델 구조 생성 및 가중치 불러오기
    try:
        model = get_model(model_name=BASE_MODEL_NAME, num_attributes=num_attributes)
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        model.to(device)
        model.eval()  # <<< 매우 중요: 모델을 평가 모드로 설정
        print(f"모델 가중치를 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        print(f"오류: 모델 가중치 파일 '{model_weights_path}'를 찾을 수 없습니다.")
        print("학습을 먼저 실행하여 모델 파일을 생성했는지 확인하세요.")
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
        for i, attr_name in enumerate(attribute_names):
            prob = probabilities[i]
            prediction = "Yes" if prob >= PREDICTION_THRESHOLD else "No"
            print(f"{attr_name:<15s} : {prediction:<5s} (신뢰도: {prob * 100:.2f}%)")
        print("-----------------")
