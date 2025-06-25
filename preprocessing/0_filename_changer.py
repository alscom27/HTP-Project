# 데이터셋의 한글 파일명을 규칙에 따라 영문으로 변경하는 스크립트

import os


def rename_files_recursively(root_directory):
    """
    지정된 루트 디렉토리와 모든 하위 디렉토리를 순회하며
    파일명을 규칙에 따라 영문으로 변경합니다.

    - 객체: 남자사람(men), 여자사람(women), 나무(tree), 집(house)
    - 메타데이터: 남(male), 여(female)

    Args:
        root_directory (str): 검색을 시작할 최상위 폴더 경로
    """
    # 이름 변경 규칙 (긴 단어를 먼저 배치하여 중복 변경 방지)
    rename_rules = {
        "남자사람": "men",
        "여자사람": "women",
        "나무": "tree",
        "집": "house",
        "남": "male",
        "여": "female",
    }

    total_changed_count = 0
    total_checked_count = 0

    try:
        # os.walk를 사용하여 루트 디렉토리부터 모든 하위 디렉토리를 순회
        for dirpath, _, filenames in os.walk(root_directory):
            print(f"--- 폴더 확인 중: {dirpath} ---")

            if not filenames:
                print("파일이 없습니다.")
                continue

            for filename in filenames:
                total_checked_count += 1
                new_filename = filename
                file_changed = False

                # 규칙을 순회하며 파일명에 키워드가 있는지 확인하고 변경
                for kor_keyword, eng_keyword in rename_rules.items():
                    if kor_keyword in new_filename:
                        new_filename = new_filename.replace(kor_keyword, eng_keyword)
                        file_changed = True

                # 파일명이 변경된 경우에만 파일 이름 변경 실행
                if file_changed:
                    old_file_path = os.path.join(dirpath, filename)
                    new_file_path = os.path.join(dirpath, new_filename)

                    # 만약 변경 후 파일명이 이미 존재한다면 중복을 피하기 위해 건너뜀
                    if os.path.exists(new_file_path):
                        print(
                            f"경고: 변경하려는 파일명 '{new_filename}'이 이미 존재하여 건너뜁니다. (경로: {dirpath})"
                        )
                        continue

                    os.rename(old_file_path, new_file_path)
                    print(f"'{filename}' -> '{new_filename}'으로 변경되었습니다.")
                    total_changed_count += 1

        print("\n=====================================")
        print(f"총 {total_checked_count}개의 파일을 확인했습니다.")
        print(f"총 {total_changed_count}개의 파일명이 변경되었습니다.")
        print("=====================================")

    except FileNotFoundError:
        print(f"오류: '{root_directory}' 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")


# --- 사용 예시 ---
# 이미지가 저장된 최상위 폴더 경로를 입력하세요.
# 이 폴더 안에 있는 모든 하위 폴더의 파일들이 변경 대상이 됩니다.
# 예: r"C:\Users\Username\Desktop\yolo_dataset" (Windows)
# 예: "/home/user/yolo_dataset" (Linux/MacOS)
TARGET_DIRECTORY = "YOUR_TOP_LEVEL_FOLDER_PATH"

rename_files_recursively(TARGET_DIRECTORY)
