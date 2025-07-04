import os
import random
import shutil
import sys

# --- 설정 ---
# 현재 파일이 있는 디렉토리 기준으로 base_dir 설정
base_dir = os.getcwd() + "/test_data/all_data"  # 스크립트가 실행되는 현재 경로를 사용
image_dir = os.path.join(base_dir, "images")
label_dir = os.path.join(base_dir, "labels")

# 소스 폴더
train_image_dir = os.path.join(image_dir, "train")
train_label_dir = os.path.join(label_dir, "train")

# 목적지 폴더
val_image_dir = os.path.join(image_dir, "val")
val_label_dir = os.path.join(label_dir, "val")

# 분할 비율
val_split_ratio = 0.2
image_extension = ".jpg"

print("\n📦 데이터 분할을 시작합니다...")

# 1. 폴더 생성
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 2. 이미지 리스트 수집
try:
    all_images = [
        f for f in os.listdir(train_image_dir) if f.lower().endswith(image_extension)
    ]
    if not all_images:
        print(f"❌ 오류: '{train_image_dir}'에 이미지가 없음.")
        sys.exit(1)
except FileNotFoundError:
    print(f"❌ 오류: '{train_image_dir}' 경로 없음.")
    sys.exit(1)

# 3. 무작위 셔플
random.shuffle(all_images)
print(f"📸 총 {len(all_images)}개의 이미지 파일을 찾음")

# 4. 분할
num_val = int(len(all_images) * val_split_ratio)
val_files = all_images[:num_val]
num_train = len(all_images) - num_val

print(f"🔀 Train: {num_train}개 | Val: {num_val}개")

# 5. 이동
moved = 0
skipped = 0
for filename in val_files:
    base, ext = os.path.splitext(filename)
    label_file = base + ".txt"

    src_img = os.path.join(train_image_dir, filename)
    dst_img = os.path.join(val_image_dir, filename)

    src_lbl = os.path.join(train_label_dir, label_file)
    dst_lbl = os.path.join(val_label_dir, label_file)

    try:
        shutil.move(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)
        else:
            print(f"⚠️ 라벨 없음: {label_file}")
        moved += 1
    except Exception as e:
        print(f"❌ 이동 실패: {filename} - {e}")
        skipped += 1

print(f"\n✅ 완료! {moved}개 이동됨, {skipped}개 실패")
