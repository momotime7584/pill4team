# evaluate.py

import torch
import argparse
import importlib
from torch.utils.data import DataLoader
from utils.data_utils import collate_fn

from data.dataset import PillDataset
from data.transforms import get_transform
from models.faster_rcnn import create_faster_rcnn_model
from engine.evaluator import evaluate # 손실과 mAP를 모두 반환

def main():
    # 1. Argument Parser 설정
    parser = argparse.ArgumentParser(description="훈련된 모델의 mAP를 평가합니다.")
    parser.add_argument(
        '-c', '--checkpoint', # 인자 이름
        type=str,# 인자의 타입
        required=True,# 이 인자는 필수임
        help="평가할 모델의 .pth 체크포인트 파일 경로. (필수)"# 도움말 메시지
    )
    # config 파일을 인자로 받아, 다른 실험 설정을 쉽게 테스트할 수 있도록 함
    parser.add_argument(
        '--config',
        type=str,
        default='configs.base_config', # 기본 설정 파일
        help="사용할 설정 파일의 파이썬 경로. (예: configs.base_config)"
    )
    args = parser.parse_args()

    # 2. 설정 파일 동적 로드
    try:
        cfg = importlib.import_module(args.config)
    except ImportError:
        print(f"오류: '{args.config}' 설정 파일을 찾을 수 없습니다.")
        return

    # 3. 설정 값 사용 (모든 값을 config 파일에서 가져옴)
    DEVICE = torch.device(cfg.DEVICE)
    BATCH_SIZE = cfg.BATCH_SIZE # BATCH_SIZE도 config에서 가져옴
    NUM_WORKERS = cfg.NUM_WORKERS
    ROOT_DIRECTORY = cfg.ROOT_DIRECTORY

    print(f"평가할 모델: {args.checkpoint}")
    print(f"사용할 설정: {args.config}")
    print(f"사용할 장치: {DEVICE}")

    # --- 데이터셋 준비 ---
    # 평가 시에는 데이터 증강을 하지 않으므로 train=False
    dataset_eval = PillDataset(root=ROOT_DIRECTORY, transforms=get_transform(train=False))
    num_classes = dataset_eval.get_num_classes()
    
    # 평가 시에는 전체 데이터셋을 사용
    # 이 부분은 실제 데이터 구조에 따라 달라질 수 있습니다.
    # 여기서는 임시로 전체 데이터셋을 사용하지만, 보통은 별도의 테스트셋을 사용합니다.
    # TODO: 테스트 데이터셋으로 바꾸기
    data_loader_eval = DataLoader(
        dataset_eval, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn
    )

    # --- 모델 로드 ---
    # TODO: configs 반영해서 생성하기
    model = create_faster_rcnn_model(num_classes)
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    except FileNotFoundError:
        print(f"오류: 체크포인트 파일을 찾을 수 없습니다 - {args.checkpoint}")
        return
    model.to(DEVICE)

    # --- 평가 실행 ---
    # evaluator는 mAP 결과만 반환
    model.eval() # 평가 시작 전에 한번만 호출
    _, map_results = evaluate(model, data_loader_eval, DEVICE)

    # --- 최종 결과 출력 ---
    print("\n--- 최종 평가 결과 ---")
    print(f"  mAP@0.5 (대회 기준): {map_results['map_50']:.4f}") # IoU 0.50 에서의 mAP
    print(f"  mAP (전체): {map_results['map']:.4f}")
    print(f"  mAP (small): {map_results['map_small']:.4f}")
    print(f"  mAP (medium): {map_results['map_medium']:.4f}")
    print(f"  mAP (large): {map_results['map_large']:.4f}")


if __name__ == '__main__':
    main()
