import os

txt_dir_val = r"test_data\labels\val"
img_dir_val = r"test_data\images\val"
txt_dir_train = r"test_data\labels\train"
img_dir_train = r"test_data\images\train"

# 확장자 제거한 파일명 세트 생성
txt_names_val = {
    os.path.splitext(f)[0] for f in os.listdir(txt_dir_val) if f.endswith(".txt")
}
img_names_val = {
    os.path.splitext(f)[0] for f in os.listdir(img_dir_val) if f.endswith(".jpg")
}
txt_names_train = {
    os.path.splitext(f)[0] for f in os.listdir(txt_dir_train) if f.endswith(".txt")
}
img_names_train = {
    os.path.splitext(f)[0] for f in os.listdir(img_dir_train) if f.endswith(".jpg")
}

# 완전 일치 여부 확인
if txt_names_val == img_names_val:
    print("✅ 모든 val 데이터 파일명이 정확히 일치해")
else:
    only_txt = sorted(txt_names_val - img_names_val)
    only_png = sorted(img_names_val - txt_names_val)
    print("❌ 파일명이 일치하지 않는 항목이 있어")
    if only_txt:
        print(f"  • TXT만 있는 파일 ({len(only_txt)}개): {only_txt}")
    if only_png:
        print(f"  • PNG만 있는 파일 ({len(only_png)}개): {only_png}")
if txt_names_train == img_names_train:
    print("✅ 모든 train 데이터 파일명이 정확히 일치해")
else:
    only_txt = sorted(txt_names_train - img_names_train)
    only_png = sorted(img_names_train - txt_names_train)
    print("❌ 파일명이 일치하지 않는 항목이 있어")
    if only_txt:
        print(f"  • TXT만 있는 파일 ({len(only_txt)}개): {only_txt}")
    if only_png:
        print(f"  • PNG만 있는 파일 ({len(only_png)}개): {only_png}")
