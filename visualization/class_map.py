from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("best.pt")

    # 모델 검증 실행
    results = model.val(
        data="test_data/4000_all_box_add_400_tree/data.yaml",
        save=True,
        save_json=True,
        save_txt=True,
    )

    # 추출할 클래스 지정
    target_classes = ["house", "person", "tree"]

    # 각 클래스의 AP 값을 저장할 리스트
    target_aps_50 = []
    target_aps_50_95 = []

    # 클래스 이름과 인덱스 정보 가져오기
    class_names = results.names

    # 클래스 인덱스를 키로, AP50 값을 값으로 하는 딕셔너리 생성
    # results.box.ap_class_index는 AP50 값이 계산된 클래스의 인덱스 리스트입니다.
    ap50_map = {
        idx: ap for idx, ap in zip(results.box.ap_class_index, results.box.ap50)
    }

    # 각 클래스별로 AP 값 추출
    for i, name in class_names.items():
        if name in target_classes:
            # mAP50-95 값 (기존 방식)
            ap50_95 = results.box.maps[i]

            # mAP50 값 (생성한 딕셔너리에서 조회)
            # .get(i, 0.0)을 사용하여 해당 클래스의 예측이 없어 값이 없는 경우 0으로 처리
            ap50 = ap50_map.get(i, 0.0)

            print(
                f"Class '{name}' (index: {i}) - AP@50: {ap50:.4f}, AP@50-95: {ap50_95:.4f}"
            )

            target_aps_50.append(ap50)
            target_aps_50_95.append(ap50_95)

    # 선택된 클래스들의 mAP 계산 및 출력
    if target_aps_50:
        specific_mAP_50 = sum(target_aps_50) / len(target_aps_50)
        print(f"\n=> mAP@50 for {target_classes}: {specific_mAP_50:.4f}")

    if target_aps_50_95:
        specific_mAP_50_95 = sum(target_aps_50_95) / len(target_aps_50_95)
        print(f"=> mAP@50-95 for {target_classes}: {specific_mAP_50_95:.4f}")

    if not target_aps_50 and not target_aps_50_95:
        print("Target classes not found in the results.")
