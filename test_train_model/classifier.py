# 이 스크립트는 나무의 속성(branch, root, crown, fruit, gnarl)을
# 분류하기 위한 딥러닝 모델을 학습하는 스크립트입니다.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

import pandas as pd
from PIL import Image
import os
import ast  # 문자열로 된 리스트를 실제 리스트로 안전하게 변환하기 위해 사용
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy


# --- 1. 설정 (Configuration) ---
class CFG:
    # 경로 설정
    ROOT_DIR = "classifier_dataset"
    CSV_PATH = os.path.join(ROOT_DIR, "classifier_labels.csv")
    IMAGE_DIR = os.path.join(ROOT_DIR, "cropped_images")

    # 모델 및 학습 하이퍼파라미터
    MODEL_NAME = "mobilenet_v3_small"  # 또는 'resnet18', 'efficientnet_b0' 등
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    NUM_ATTRIBUTES = 5  # branch, root, crown, fruit, gnarl

    # 데이터 분할 비율
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42


# --- 2. 커스텀 데이터셋 클래스 정의 ---
class TreeAttributesDataset(Dataset):
    """
    CSV 파일과 이미지 폴더로부터 데이터를 읽어오는 커스텀 데이터셋
    """

    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # CSV 파일에서 이미지의 상대 경로를 가져옵니다. (예: 'cropped_images/0_0.png')
        img_relative_path = self.dataframe.iloc[idx, 0]

        # 데이터셋의 루트 디렉토리와 상대 경로를 합쳐 전체 경로를 만듭니다.
        # 예: 'classifier_dataset' + 'cropped_images/0_0.png' -> 'classifier_dataset/cropped_images/0_0.png'
        full_img_path = os.path.join(self.root_dir, img_relative_path)

        # 이미지 열기
        try:
            image = Image.open(full_img_path).convert("RGB")
        except FileNotFoundError:
            print(f"오류: 파일을 찾을 수 없습니다 - {full_img_path}")
            # 빈 이미지를 반환하거나, 에러를 발생시켜 Dataloader가 해당 샘플을 건너뛰게 할 수 있습니다.
            # 이 경우, None을 반환하도록 하여 DataLoader의 collate_fn에서 처리할 수 있습니다.
            # 하지만 더 간단한 방법은 데이터 준비 단계에서 경로가 확실한지 확인하는 것입니다.
            # 여기서는 에러를 다시 발생시켜 어떤 파일에 문제가 있는지 명확히 합니다.
            raise

        # 라벨 가져오기 및 파싱
        # 라벨은 "[1, 0, 1, 0, 0]" 형태의 '문자열'이므로 ast.literal_eval로 실제 리스트로 변환
        label_str = self.dataframe.iloc[idx, 1]
        label_list = ast.literal_eval(label_str)
        label_tensor = torch.FloatTensor(
            label_list
        )  # Loss 계산을 위해 FloatTensor로 변환

        # 이미지 변환 (Augmentation 등)
        if self.transform:
            image = self.transform(image)

        return image, label_tensor


# --- 3. 모델 정의 ---
def get_model(num_attributes, pretrained=True):
    """
    사전 학습된 모델을 불러와 마지막 레이어를 우리 작업에 맞게 수정
    """
    # MobileNetV3 Small 모델 불러오기
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # 마지막 분류 레이어(classifier)를 새로운 레이어로 교체
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_attributes)

    print(f"모델 '{CFG.MODEL_NAME}'을 불러왔습니다.")
    print(f"출력 레이어를 {num_attributes}개의 속성에 맞게 수정했습니다.")
    return model


# --- 4. 메인 학습 로직 ---
def main():
    # 장치 설정 (GPU 우선 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 데이터 준비
    df = pd.read_csv(CFG.CSV_PATH)
    train_df, val_df = train_test_split(
        df, test_size=CFG.VALIDATION_SPLIT, random_state=CFG.RANDOM_STATE
    )

    # 데이터 변환 (Augmentation) 정의
    train_transform = transforms.Compose(
        [
            transforms.Resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
    train_dataset = TreeAttributesDataset(
        train_df, CFG.ROOT_DIR, transform=train_transform
    )
    val_dataset = TreeAttributesDataset(val_df, CFG.ROOT_DIR, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 모델, 손실 함수, 옵티마이저 정의
    model = get_model(num_attributes=CFG.NUM_ATTRIBUTES).to(device)

    # 다중 레이블 분류에는 BCEWithLogitsLoss를 사용하는 것이 표준
    # 이 함수는 Sigmoid 활성화 함수를 내장하고 있어 수치적으로 안정적임
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE)

    # 학습 루프
    best_val_loss = float("inf")
    best_model_weights = None

    for epoch in range(CFG.NUM_EPOCHS):
        # --- 학습(Train) ---
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

        # --- 검증(Validation) ---
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

                # 정확도 계산 (0.5를 임계값으로 사용)
                preds = torch.sigmoid(outputs) > 0.5
                val_corrects += (preds == labels).all(dim=1).sum().item()
                total_samples += labels.size(0)

        # 에포크 결과 출력
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (val_corrects / total_samples) * 100

        print(
            f"\nEpoch {epoch + 1}/{CFG.NUM_EPOCHS} -> "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

        # 최고의 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"*** Best model updated (val_loss: {best_val_loss:.4f}) ***")

    # 학습 완료 후 최고의 가중치를 모델에 로드하고 저장
    if best_model_weights:
        model.load_state_dict(best_model_weights)
        torch.save(model.state_dict(), f"best_{CFG.MODEL_NAME}_model.pth")
        print(
            f"\n학습 완료! 최고의 모델이 'best_{CFG.MODEL_NAME}_model.pth' 파일로 저장되었습니다."
        )


if __name__ == "__main__":
    main()
