# CSV 파일의 첫 번째 열에 있는 영문 이름을 한글로 변경하는 스크립트입니다.
# 지정된 폴더에 있는 CSV 파일을 읽고, 첫 번째 열의 이름을 한글로 변경한 후,
# 새로운 CSV 파일로 저장합니다.

import os
import csv

# 1. CSV 파일이 있는 폴더 경로와 파일 이름을 설정
# 예: 'C:/Users/Test/Documents'
TARGET_DIR = "C:/Users/UserK/Desktop/HTP-Project/labeling_result"  # <-- 여기에 CSV 파일이 있는 폴더 경로를 입력
CSV_FILENAME = "tree_labeling_result.csv"  # <-- 여기에 원본 CSV 파일 이름을 입력


# 이름을 변경할 규칙 (영문 -> 한글)
rename_map = {
    "men": "남자사람",
    "women": "여자사람",
    "tree": "나무",
    "house": "집",
    "male": "남",
    "female": "여",
}

# 원본 CSV 파일과 새로 저장할 CSV 파일의 전체 경로 설정
original_csv_path = os.path.join(TARGET_DIR, CSV_FILENAME)
# 새로운 파일 이름에 '_modified'를 붙여 저장
new_csv_path = os.path.join(
    TARGET_DIR, f"{os.path.splitext(CSV_FILENAME)[0]}_modified.csv"
)

print(f"CSV 파일 처리 시작: '{original_csv_path}'")
print(f"결과 저장 파일: '{new_csv_path}'\n")

try:
    # 수정된 내용을 담을 리스트
    modified_rows = []

    # 원본 CSV 파일을 읽음 (한글 깨짐 방지)
    with open(original_csv_path, mode="r", newline="", encoding="utf-8-sig") as infile:
        reader = csv.reader(infile)

        for row in reader:
            if not row:
                modified_rows.append([])  # 빈 줄도 그대로 유지
                continue

            # 첫 번째 열의 파일 이름 문자열을 가져옴
            original_text = row[0].strip()
            base_name, extension = os.path.splitext(original_text)

            # 파일 이름에서 확장자를 제외한 부분을 언더스코어로 분리
            parts = base_name.split("_")

            # 각 부분을 rename_map을 사용하여 한글로 변경
            new_parts = [rename_map.get(part, part) for part in parts]

            # 변경된 부분을 다시 언더스코어로 합침
            new_base_name = "_".join(new_parts)

            # 수정된 텍스트를 생성
            modified_text = f"{new_base_name}{extension}"

            # 원본 row의 첫 번째 열을 수정된 텍스트로 교체
            # (만약 CSV에 여러 열이 있더라도 나머지 데이터는 보존)
            new_row = [modified_text] + row[1:]
            modified_rows.append(new_row)

            # 변경 내용 출력
            if original_text != modified_text:
                print(f"🔄 변경: '{original_text}' -> '{modified_text}'")
            else:
                print(f"➡️ 유지: '{original_text}'")

    # 수정된 내용을 새로운 CSV 파일에 씀
    with open(new_csv_path, mode="w", newline="", encoding="utf-8-sig") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(modified_rows)

    print(f"\n✅ 작업 완료! 결과가 '{new_csv_path}' 파일에 저장되었습니다.")

except FileNotFoundError:
    print(
        f"❌ 오류: 원본 CSV 파일('{original_csv_path}')을 찾을 수 없습니다. 경로를 확인해주세요."
    )
except Exception as e:
    print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")
