# 이 스크립트는 학습된 통합 모델의 성능을 종합적으로 평가합니다.
# 'classifier_labels.csv'에 있는 전체 데이터에 대한 정확도를 계산하고,
# 객체 라벨(tree, house, person)별 성능을 나누어 상세히 분석합니다.

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
    classification_report,
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
    # 다른 모델 아키텍처를 사용하는 경우 여기에 추가
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=None)
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
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(
                f"경고: 파일을 찾을 수 없습니다 - {img_path}. 검은 이미지로 대체합니다."
            )
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # === [수정된 부분 시작] ===
        # 현재 행(row)을 모델이 학습한 전체 속성 컬럼 기준으로 재정렬합니다.
        # 만약 현재 행에 존재하지 않는 속성 컬럼이 있다면, 해당 값은 0으로 채워집니다.
        # 이를 통해 KeyError를 방지하고 데이터 구조의 일관성을 유지합니다.
        label_series = row.reindex(self.attribute_columns).fillna(0)
        labels = torch.FloatTensor(label_series.values.astype(float))
        # === [수정된 부분 끝] ===

        if self.transform:
            image = self.transform(image)

        return image, labels


# --- 3. 평가 결과 출력 함수 ---
def print_evaluation_results(y_true, y_pred, attribute_names, title):
    """
    주어진 예측과 실제 라벨을 바탕으로 평가 지표를 계산하고 출력
    """
    print("\n" + "=" * 60)
    print(f"    {title}")
    print("=" * 60)
    print(f"총 평가 이미지 수: {len(y_true)}개")

    if len(y_true) == 0:
        print("평가할 데이터가 없습니다.")
        return

    # 1. 전체 정확도 (Overall Accuracy)
    overall_accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
    print(f"\n[1] 전체 정확도 (Overall Accuracy)")
    print(f"    - 모든 속성의 개별 예측 중 정답 비율: {overall_accuracy * 100:.2f}%")

    # 2. 완전 일치 비율 (Exact Match Ratio)
    exact_match_ratio = np.all(y_true == y_pred, axis=1).mean()
    print(f"\n[2] 완전 일치 비율 (Exact Match Ratio)")
    print(
        f"    - 한 이미지의 모든 속성을 완벽하게 맞춘 비율: {exact_match_ratio * 100:.2f}%"
    )

    # 3. 속성별 상세 지표 (Per-Attribute Metrics)
    print("\n[3] 속성별 상세 지표")
    report = classification_report(
        y_true, y_pred, target_names=attribute_names, zero_division=0
    )
    print(report)
    print("=" * 60)


# --- 4. 메인 평가 로직 ---
def evaluate_model():
    # ===================================================================
    #                           사용자 설정
    # ===================================================================
    BASE_MODEL_NAME = "efficientnet_b2"
    MODEL_WEIGHTS_PATH = f"best_unified_model_{BASE_MODEL_NAME}.pth"

    IMAGE_SIZE = 260
    BATCH_SIZE = 16
    PREDICTION_THRESHOLD = 0.5

    ROOT_DIR = "classifier_dataset"
    CSV_PATH = os.path.join(ROOT_DIR, "classifier_labels.csv")
    # ===================================================================

    # --- 준비 단계 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    print(f"평가 모델: '{MODEL_WEIGHTS_PATH}'")

    # 모델 및 속성 정보 불러오기
    try:
        checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
        attribute_columns = checkpoint["attribute_columns"]
        num_attributes = len(attribute_columns)

        model = get_model(model_name=BASE_MODEL_NAME, num_attributes=num_attributes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        print("모델 가중치를 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        print(f"오류: 모델 가중치 파일 '{MODEL_WEIGHTS_PATH}'를 찾을 수 없습니다.")
        return

    # 데이터 로딩
    try:
        df_full = pd.read_csv(CSV_PATH)
        # Person 라벨 통합 ('men', 'women' -> 'person')
        df_full["label"] = df_full["label"].replace(["men", "women"], "person")
    except FileNotFoundError:
        print(f"오류: 라벨 파일 '{CSV_PATH}'를 찾을 수 없습니다.")
        return

    # 이미지 변환 정의
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 데이터셋 및 데이터로더 생성 (전체 데이터 대상)
    full_dataset = EvaluationDataset(
        df_full, ROOT_DIR, attribute_columns, transform=eval_transform
    )
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 전체 예측 및 정답 수집
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(full_loader, desc="전체 모델 예측 중"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > PREDICTION_THRESHOLD
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # --- 전체 성능 평가 ---
    print_evaluation_results(
        all_labels, all_preds, attribute_columns, "전체 데이터셋 성능 평가"
    )

    # --- 객체 라벨별 성능 평가 ---
    df_full_preds = pd.DataFrame(all_preds, columns=attribute_columns)
    df_full_labels = pd.DataFrame(all_labels, columns=attribute_columns)

    for label_type in sorted(df_full["label"].unique()):
        label_indices = df_full[df_full["label"] == label_type].index

        # 라벨 인덱스를 사용하여 예측값과 실제값 추출
        y_pred_subset = df_full_preds.loc[label_indices].values
        y_true_subset = df_full_labels.loc[label_indices].values

        # 해당 라벨과 관련된 속성만 필터링
        # (예: 'house' 평가 시 'door_yn', 'roof_yn' 등만 포함)
        relevant_attrs_indices = [
            i
            for i, col in enumerate(attribute_columns)
            if np.any(y_true_subset[:, i] > 0)
        ]
        relevant_attr_names = [attribute_columns[i] for i in relevant_attrs_indices]

        if not relevant_attr_names:
            print(
                f"\n'{label_type}' 라벨에 해당하는 활성화된 속성이 없어 평가를 건너뜁니다."
            )
            continue

        y_pred_filtered = y_pred_subset[:, relevant_attrs_indices]
        y_true_filtered = y_true_subset[:, relevant_attrs_indices]

        print_evaluation_results(
            y_true_filtered,
            y_pred_filtered,
            relevant_attr_names,
            f"'{label_type}' 객체 성능 평가",
        )


if __name__ == "__main__":
    evaluate_model()
