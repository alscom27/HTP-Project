# 이 스크립트는 'classifier_labels.csv' 파일을 기반으로
# 다양한 객체(예: 나무, 집, 사람)의 모든 속성을 한 번에 분류하는
# 딥러닝 모델을 학습하는 통합 스크립트입니다.

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

    # 모델 및 학습 하이퍼파라미터
    MODEL_NAME = "efficientnet_b0"  # 또는 'mobilenet_v3_small', 'resnet18' 등
    IMAGE_SIZE = 224
    BATCH_SIZE = 16  # GPU 메모리에 따라 조정 가능
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 30
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 분할 비율
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42


# --- 2. 커스텀 데이터셋 클래스 정의 ---
class UnifiedAttributeDataset(Dataset):
    """
    통합된 CSV 파일과 이미지 폴더로부터 데이터를 읽어오는 커스텀 데이터셋
    """

    def __init__(self, dataframe, root_dir, attribute_columns, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.attribute_columns = attribute_columns
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_relative_path = self.dataframe.iloc[idx]["image_path"]
        full_img_path = os.path.join(self.root_dir, img_relative_path)

        try:
            image = Image.open(full_img_path).convert("RGB")
        except FileNotFoundError:
            print(f"오류: 파일을 찾을 수 없습니다 - {full_img_path}")
            # 임시로 검은색 이미지를 반환하여 학습 중단을 방지
            return torch.zeros((3, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)), torch.zeros(
                len(self.attribute_columns)
            )

        label_values = self.dataframe.loc[idx, self.attribute_columns].values
        label_tensor = torch.FloatTensor(label_values.astype(float))

        if self.transform:
            image = self.transform(image)

        return image, label_tensor


# --- 3. 모델 정의 ---
def get_model(num_attributes, pretrained=True):
    """
    사전 학습된 모델을 불러와 마지막 레이어를 우리 작업에 맞게 수정
    """
    model = None
    weights = None
    if pretrained:
        if CFG.MODEL_NAME == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        # elif CFG.MODEL_NAME == "mobilenet_v3_small":
        #     weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1

    if CFG.MODEL_NAME == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_attributes)
    # elif CFG.MODEL_NAME == "mobilenet_v3_small":
    #     model = models.mobilenet_v3_small(weights=weights)
    #     in_features = model.classifier[-1].in_features
    #     model.classifier[-1] = nn.Linear(in_features, num_attributes)
    else:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {CFG.MODEL_NAME}")

    print(f"모델 '{CFG.MODEL_NAME}'을 불러왔습니다. (Pretrained: {pretrained})")
    print(f"출력 레이어를 {num_attributes}개의 속성에 맞게 수정했습니다.")
    return model


# --- 4. 메인 학습 로직 ---
def main():
    print(f"사용 장치: {CFG.DEVICE}")
    print("모든 객체의 속성을 통합하여 모델 학습을 시작합니다.")

    # 데이터 준비
    try:
        df_full = pd.read_csv(CFG.CSV_PATH)
    except FileNotFoundError:
        print(f"오류: CSV 파일 '{CFG.CSV_PATH}'를 찾을 수 없습니다.")
        print(
            "6번 스크립트를 먼저 실행하여 'classifier_labels.csv' 파일을 생성했는지 확인해주세요."
        )
        return

    # 'image_path'와 'label'을 제외한 모든 컬럼을 속성으로 간주
    attribute_columns = sorted(
        [col for col in df_full.columns if col not in ["image_path", "label"]]
    )
    num_attributes = len(attribute_columns)

    if num_attributes == 0:
        print("오류: 학습할 속성 컬럼이 CSV 파일에 없습니다.")
        return

    print(f"학습 대상 속성 ({num_attributes}개): {attribute_columns}")

    # Person 라벨 통합 ('men', 'women' -> 'person')
    df_full["label"] = df_full["label"].replace(["men", "women"], "person")

    # 데이터 분할 (Stratify by label to maintain distribution)
    train_df, val_df = train_test_split(
        df_full,
        test_size=CFG.VALIDATION_SPLIT,
        random_state=CFG.RANDOM_STATE,
        stratify=df_full["label"],  # 라벨 분포를 유지하며 분할
    )

    # 데이터 변환 (Augmentation) 정의
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    }

    # Dataset 및 DataLoader 생성
    train_dataset = UnifiedAttributeDataset(
        train_df, CFG.ROOT_DIR, attribute_columns, transform=data_transforms["train"]
    )
    val_dataset = UnifiedAttributeDataset(
        val_df, CFG.ROOT_DIR, attribute_columns, transform=data_transforms["val"]
    )

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=CFG.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=CFG.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        ),
    }

    # 모델, 손실 함수, 옵티마이저 정의
    model = get_model(num_attributes=num_attributes).to(CFG.DEVICE)
    criterion = nn.BCEWithLogitsLoss()  # Multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )

    # 학습 루프
    best_val_loss = float("inf")
    best_model_weights = None

    for epoch in range(CFG.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{CFG.NUM_EPOCHS}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            for images, labels in tqdm(
                dataloaders[phase], desc=f"{phase.capitalize()}"
            ):
                images, labels = images.to(CFG.DEVICE), labels.to(CFG.DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            # Exact match accuracy
            accuracy = (all_preds == all_labels).all(dim=1).float().mean().item() * 100

            print(
                f"{phase.capitalize()} Loss: {epoch_loss:.4f} Accuracy: {accuracy:.2f}%"
            )

            if phase == "val":
                scheduler.step(epoch_loss)
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
                    print(f"*** Best model updated (val_loss: {best_val_loss:.4f}) ***")

    # 최고의 가중치를 모델에 로드하고 저장
    if best_model_weights:
        model.load_state_dict(best_model_weights)
        save_path = f"best_unified_model_{CFG.MODEL_NAME}.pth"
        # 모델 전체를 저장하여 나중에 속성 목록과 함께 사용
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "attribute_columns": attribute_columns,
            },
            save_path,
        )
        print(f"\n학습 완료! 최고의 통합 모델이 '{save_path}' 파일로 저장되었습니다.")
        print(f"저장된 모델은 다음 속성들을 예측합니다: {attribute_columns}")


if __name__ == "__main__":
    main()
