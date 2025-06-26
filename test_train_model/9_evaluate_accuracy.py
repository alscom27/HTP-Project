# 이 스크립트는 학습된 모델의 성능을 평가합니다.
# 'classifier_labels.csv'에 있는 정답과 모델의 예측을 비교하여
# 다양한 정확도 지표를 계산하고 출력합니다.

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    multilabel_confusion_matrix,
)


# --- 1. 모델 정의 (학습/예측 스크립트와 동일) ---
def get_model(model_name, num_attributes):
    """
    사전 학습된 모델 구조를 불러와 마지막 레이어를 우리 작업에 맞게 수정
    """
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_attributes)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_attributes)
    else:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {model_name}")
    return model


# --- 2. 평가를 위한 커스텀 데이터셋 ---
# 학습 스크립트의 데이터셋 클래스와 유사하지만, 평가용으로 경로와 라벨만 반환
class EvaluationDataset(Dataset):
    def __init__(self, dataframe, root_dir, attribute_columns, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.attribute_columns = attribute_columns
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.root_dir, row["image_path"])
        image = Image.open(img_path).convert("RGB")

        labels = torch.FloatTensor(row[self.attribute_columns].values.astype(float))

        if self.transform:
            image = self.transform(image)

        return image, labels


# --- 3. 메인 평가 로직 ---
def evaluate_model():
    # ===================================================================
    #                           사용자 설정
    # ===================================================================
    # <<< 1. 평가하려는 객체의 종류를 입력하세요 ('tree', 'house', 'person') >>>
    TARGET_LABEL = "tree"

    # <<< 2. 평가에 사용할 모델의 이름을 입력하세요 >>>
    BASE_MODEL_NAME = "efficientnet_b0"

    # --- (고급 설정) ---
    IMAGE_SIZE = 224
    BATCH_SIZE = 16  # 평가 시에는 배치 크기를 좀 더 키워도 됩니다.
    PREDICTION_THRESHOLD = 0.5

    # --- 파일 경로 자동 설정 ---
    ROOT_DIR = "classifier_dataset"
    CSV_PATH = os.path.join(ROOT_DIR, "classifier_labels.csv")
    MODEL_WEIGHTS_PATH = f"best_{TARGET_LABEL}_{BASE_MODEL_NAME}.pth"
    # ===================================================================

    # --- 준비 단계 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    print(f"평가 대상: '{TARGET_LABEL}' | 모델 파일: '{MODEL_WEIGHTS_PATH}'")

    # 데이터 로딩 및 필터링
    try:
        df_full = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"오류: 라벨 파일 '{CSV_PATH}'를 찾을 수 없습니다.")
        return

    # 타겟 라벨에 따른 속성 컬럼 정의
    if TARGET_LABEL == "tree":
        attribute_columns = ["branch_yn", "crown_yn", "fruit_yn", "gnarl_yn", "root_yn"]
        df_eval = df_full[df_full["label"] == "tree"]
    elif TARGET_LABEL == "house":
        attribute_columns = [
            "door_yn",
            "roof_yn",
            "window_cnt_1",
            "window_cnt_2",
            "window_cnt_more_than_3",
        ]
        df_eval = df_full[df_full["label"] == "house"]
    elif TARGET_LABEL == "person":
        attribute_columns = ["eye_yn", "leg_yn", "mouth_yn"]
        df_eval = df_full[df_full["label"].isin(["men", "women"])]
    else:
        raise ValueError("지원하지 않는 TARGET_LABEL입니다.")

    num_attributes = len(attribute_columns)

    # 모델 불러오기
    try:
        model = get_model(model_name=BASE_MODEL_NAME, num_attributes=num_attributes)
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model.to(device)
        model.eval()
        print("모델 가중치를 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        print(f"오류: 모델 가중치 파일 '{MODEL_WEIGHTS_PATH}'를 찾을 수 없습니다.")
        return

    # 이미지 변환 정의 (학습 시 Validation과 동일하게)
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 데이터셋 및 데이터로더 생성
    eval_dataset = EvaluationDataset(
        df_eval, ROOT_DIR, attribute_columns, transform=eval_transform
    )
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 예측 및 정답 수집
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(eval_loader, desc="모델 예측 중"):
            images = images.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs) > PREDICTION_THRESHOLD

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # NumPy 배열로 변환
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # --- 정확도 계산 및 결과 출력 ---
    print("\n" + "=" * 50)
    print(" " * 18 + "모델 성능 평가 결과")
    print("=" * 50)
    print(f"총 평가 이미지 수: {len(df_eval)}개")

    # 1. 전체 정확도 (Overall Accuracy)
    overall_accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
    print(f"\n[1] 전체 정확도 (Overall Accuracy)")
    print(f"    - 모든 속성의 개별 예측 중 정답 비율: {overall_accuracy * 100:.2f}%")

    # 2. 완전 일치 비율 (Exact Match Ratio)
    exact_match_ratio = np.all(all_preds == all_labels, axis=1).mean()
    print(f"\n[2] 완전 일치 비율 (Exact Match Ratio)")
    print(
        f"    - 한 이미지의 모든 속성을 완벽하게 맞춘 비율: {exact_match_ratio * 100:.2f}%"
    )

    # 3. 속성별 상세 지표 (Per-Attribute Metrics)
    print("\n[3] 속성별 상세 지표")
    # precision, recall, f1-score, support 계산
    p, r, f1, s = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    # 헤더 출력
    print(
        f"    {'속성명':<25} | {'정확도(Acc)':<12} | {'정밀도(Prec)':<12} | {'재현율(Recall)':<14} | {'F1-Score':<12}"
    )
    print("    " + "-" * 80)

    for i, attr_name in enumerate(attribute_columns):
        attr_acc = accuracy_score(all_labels[:, i], all_preds[:, i])
        print(
            f"    {attr_name:<25} | {attr_acc * 100:<12.2f} | {p[i] * 100:<12.2f} | {r[i] * 100:<14.2f} | {f1[i] * 100:<12.2f}"
        )

    print("=" * 50)


if __name__ == "__main__":
    evaluate_model()
