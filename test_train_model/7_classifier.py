# 이 스크립트는 'classifier_labels.csv' 파일을 기반으로
# 특정 객체(예: 나무, 집, 사람)의 속성을 분류하기 위한
# 딥러닝 모델을 학습하는 범용 스크립트입니다.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

import pandas as pd
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy


# --- 1. 설정 (Configuration) ---
class CFG:
    # 경로 설정
    ROOT_DIR = "classifier_dataset"
    CSV_PATH = os.path.join(ROOT_DIR, "classifier_labels.csv")

    # !!!! 학습할 객체 라벨을 여기서 선택 !!!!
    # 'tree', 'house', 'person' 중 하나를 선택하세요.
    TARGET_LABEL = "tree"

    # 모델 및 학습 하이퍼파라미터
    MODEL_NAME = "mobilenet_v3_small"  # 또는 'resnet18', 'efficientnet_b0' 등
    IMAGE_SIZE = 224
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 30

    # 데이터 분할 비율
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42


# --- 2. 커스텀 데이터셋 클래스 정의 ---
class ObjectAttributeDataset(Dataset):
    """
    CSV 파일과 이미지 폴더로부터 데이터를 읽어오는 범용 커스텀 데이터셋
    """

    def __init__(self, dataframe, root_dir, attribute_columns, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)  # 인덱스 초기화
        self.root_dir = root_dir
        self.attribute_columns = attribute_columns
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # CSV 파일에서 이미지의 상대 경로를 가져옵니다. (예: 'cropped_images/tree/0_0.png')
        img_relative_path = self.dataframe.iloc[idx]["image_path"]
        full_img_path = os.path.join(self.root_dir, img_relative_path)

        try:
            image = Image.open(full_img_path).convert("RGB")
        except FileNotFoundError:
            print(f"오류: 파일을 찾을 수 없습니다 - {full_img_path}")
            raise

        # 라벨 가져오기 (DataFrame에서 해당 속성 컬럼들을 선택)
        label_values = self.dataframe.loc[idx, self.attribute_columns].values
        label_tensor = torch.FloatTensor(label_values.astype(float))

        # 이미지 변환 (Augmentation 등)
        if self.transform:
            image = self.transform(image)

        return image, label_tensor


# --- 3. 모델 정의 ---
def get_model(num_attributes, pretrained=True):
    """
    사전 학습된 모델을 불러와 마지막 레이어를 우리 작업에 맞게 수정
    """
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_attributes)

    print(f"모델 '{CFG.MODEL_NAME}'을 불러왔습니다.")
    print(f"출력 레이어를 {num_attributes}개의 속성에 맞게 수정했습니다.")
    return model


# --- 4. 메인 학습 로직 ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    print(f"'{CFG.TARGET_LABEL}' 객체에 대한 모델 학습을 시작합니다.")

    # 데이터 준비
    df_full = pd.read_csv(CFG.CSV_PATH)

    # 타겟 라벨에 따른 속성 컬럼 및 데이터프레임 필터링
    if CFG.TARGET_LABEL == "tree":
        attribute_columns = ["branch_yn", "crown_yn", "fruit_yn", "gnarl_yn", "root_yn"]
        df_filtered = df_full[df_full["label"] == "tree"]
    elif CFG.TARGET_LABEL == "house":
        attribute_columns = [
            "door_yn",
            "roof_yn",
            "window_cnt_1",
            "window_cnt_2",
            "window_cnt_more_than_3",
        ]
        df_filtered = df_full[df_full["label"] == "house"]
    elif CFG.TARGET_LABEL == "person":
        attribute_columns = ["eye_yn", "leg_yn", "mouth_yn"]
        # 'men'과 'women' 라벨을 모두 'person'으로 간주하여 필터링
        df_filtered = df_full[df_full["label"].isin(["men", "women"])]
    else:
        raise ValueError(f"지원하지 않는 TARGET_LABEL입니다: {CFG.TARGET_LABEL}")

    num_attributes = len(attribute_columns)
    print(f"학습 대상 속성 ({num_attributes}개): {attribute_columns}")

    if df_filtered.empty:
        print(
            f"오류: CSV 파일에서 '{CFG.TARGET_LABEL}' 라벨을 가진 데이터를 찾을 수 없습니다."
        )
        return

    train_df, val_df = train_test_split(
        df_filtered, test_size=CFG.VALIDATION_SPLIT, random_state=CFG.RANDOM_STATE
    )

    # 데이터 변환 (Augmentation) 정의
    train_transform = transforms.Compose(
        [
            transforms.Resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Dataset 및 DataLoader 생성
    train_dataset = ObjectAttributeDataset(
        train_df, CFG.ROOT_DIR, attribute_columns, transform=train_transform
    )
    val_dataset = ObjectAttributeDataset(
        val_df, CFG.ROOT_DIR, attribute_columns, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 모델, 손실 함수, 옵티마이저 정의
    model = get_model(num_attributes=num_attributes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE)

    # 학습 루프
    best_val_loss = float("inf")
    best_model_weights = None

    for epoch in range(CFG.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{CFG.NUM_EPOCHS} [Train]"
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{CFG.NUM_EPOCHS} [Val]"
            ):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                val_corrects += (preds == labels.bool()).all(dim=1).sum().item()
                total_samples += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (val_corrects / total_samples) * 100

        print(
            f"\nEpoch {epoch + 1}/{CFG.NUM_EPOCHS} -> "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"*** Best model updated (val_loss: {best_val_loss:.4f}) ***")

    # 최고의 가중치를 모델에 로드하고 저장
    if best_model_weights:
        model.load_state_dict(best_model_weights)
        save_path = f"best_{CFG.TARGET_LABEL}_{CFG.MODEL_NAME}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"\n학습 완료! 최고의 모델이 '{save_path}' 파일로 저장되었습니다.")


if __name__ == "__main__":
    main()
