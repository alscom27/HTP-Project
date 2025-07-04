import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# -------------------------------------------------------------------
#                                한글 폰트 설정
# -------------------------------------------------------------------
# 차트의 한글 제목이 깨지지 않도록 주석을 해제하고 사용하세요.
try:
    font_name = font_manager.FontProperties(
        fname="c:/Windows/Fonts/malgun.ttf"
    ).get_name()
    rc("font", family=font_name)
    print("✅ 한글 폰트(맑은 고딕) 설정 완료.")
except FileNotFoundError:
    print("⚠️ 한글 폰트(맑은 고딕)를 찾을 수 없습니다. 차트의 한글이 깨질 수 있습니다.")
plt.rcParams["axes.unicode_minus"] = False


# -------------------------------------------------------------------
#                           시각화 함수 (기존과 동일)
# -------------------------------------------------------------------
def create_donut_chart(data, column_name, title, output_dir):
    """지정된 컬럼의 데이터를 바탕으로 도넛 차트를 생성하고 저장합니다."""
    if column_name not in data.columns or data[column_name].isnull().all():
        print(f"'{column_name}'에 대한 데이터가 없어 차트를 생성하지 않습니다.")
        return

    counts = data[column_name].value_counts()
    if all(item in counts.index for item in ["y", "n"]):
        counts = counts.reindex(["y", "n"])
    colors = ["#9fa8da", "#5c6bc0"]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect="equal"))

    def func_autopct(pct):
        return f"{int(round(pct * sum(counts.values()) / 100.0))}"

    wedges, texts, autotexts = ax.pie(
        counts,
        autopct=func_autopct,
        startangle=90,
        colors=colors,
        textprops=dict(color="w", weight="bold", size=16),
        pctdistance=0.8,
    )
    centre_circle = plt.Circle((0, 0), 0.65, fc="white")
    fig.gca().add_artist(centre_circle)
    ax.legend(
        wedges,
        [f"있음 (y)", f"없음 (n)"],
        title="범주",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=12,
    )
    ax.set_title(title, fontsize=20, pad=20)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{column_name}_ratio.png")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ '{title}' 차트 저장 완료: {output_path}")


# ===================================================================
#                                메인 스크립트
# ===================================================================

print("--- 스크립트 실행 시작 ---")

# 1. ❗ 사용자 설정 ❗
# 절대 경로를 사용하여 어떤 위치에서 실행해도 동일하게 작동하도록 합니다.
BASE_PROJECT_DIR = r"C:\Users\main\Desktop\HTP-Project"

IMAGE_DIR = os.path.join(BASE_PROJECT_DIR, "visualization", "images2", "images")
LABEL_DIR = os.path.join(BASE_PROJECT_DIR, "visualization", "images2", "labels")
OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, "result")

# 클래스 정보 설정
CLASS_MAPPING = {11: "eye", 12: "leg", 13: "mouth", 14: "arm"}

# --- 경로 존재 여부 확인 (디버깅) ---
print(f"이미지 폴더 확인 중: {IMAGE_DIR}")
if not os.path.exists(IMAGE_DIR):
    print(f"❌ ERROR: 이미지 폴더를 찾을 수 없습니다! 경로와 오타를 확인하세요.")
    exit()

print(f"라벨 폴더 확인 중: {LABEL_DIR}")
if not os.path.exists(LABEL_DIR):
    print(f"❌ ERROR: 라벨 폴더를 찾을 수 없습니다! 경로와 오타를 확인하세요.")
    exit()

# 2. 스크립트 실행 준비
os.makedirs(OUTPUT_DIR, exist_ok=True)
results_list = []
target_class_names = list(CLASS_MAPPING.values())
image_files = [
    f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

print(f"✅ 총 {len(image_files)}개의 이미지 파일을 찾았습니다.")
if not image_files:
    print("⚠️ 처리할 이미지가 없습니다. 폴더가 비어있는지 확인해주세요.")
    exit()

# 3. 각 이미지의 라벨 파일 분석 (이하 기존과 동일)
print("\n라벨 파일 분석을 시작합니다...")
for i, image_file in enumerate(image_files):
    if (i + 1) % 100 == 0:
        print(f"  ... {i+1}/{len(image_files)} 처리 중 ...")

    record = {"filename": image_file}
    for name in target_class_names:
        record[f"{name}_yn"] = "n"

    base_filename = os.path.splitext(image_file)[0]
    label_path = os.path.join(LABEL_DIR, f"{base_filename}.txt")

    if os.path.exists(label_path):
        found_indices = set()
        with open(label_path, "r") as f:
            for line in f:
                try:
                    class_index = int(line.split()[0])
                    found_indices.add(class_index)
                except (ValueError, IndexError):
                    continue
        for index, name in CLASS_MAPPING.items():
            if index in found_indices:
                record[f"{name}_yn"] = "y"

    results_list.append(record)

print("✅ 모든 라벨 파일 분석 완료!")

# 4. 데이터프레임 변환 및 CSV 저장
df = pd.DataFrame(results_list)
csv_path = os.path.join(OUTPUT_DIR, "label_analysis.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"✅ 분석 결과가 CSV 파일로 저장되었습니다: {csv_path}")

# 5. 시각화
print("\n결과 시각화를 시작합니다...")
for name in target_class_names:
    column_name = f"{name}_yn"
    chart_title = f"'{name.capitalize()}' 라벨 유무 비율"
    create_donut_chart(df, column_name, chart_title, OUTPUT_DIR)

print("\n🎉 모든 작업이 완료되었습니다!")
