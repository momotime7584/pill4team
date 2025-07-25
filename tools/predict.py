# tools/predict.py

import torch
import argparse
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from PIL import Image
import glob, os
# 로컬 모듈 임포트
from configs import base_config as cfg
from data.dataset import PillDataset # '매핑 정보'를 얻기 위해 임포트
# from models.faster_rcnn import create_faster_rcnn_model
from models import create_model

from data.transforms import get_transform

# 테스트 데이터셋을 위한 간단한 클래스 (라벨이 없음)
class PillTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transforms=None):
        self.image_paths = image_paths
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            # img, _ = self.transforms(image=img, labels=None) # 타겟이 없으므로 None 전달
            transformed = self.transforms(image=image, boxes=None, labels=None) # 타겟이 없으므로 None 전달
            img = transformed['image']

        return img, os.path.basename(img_path)

    def __len__(self):
        return len(self.image_paths)

def main():
    parser = argparse.ArgumentParser(description="모델로 예측을 수행하고 제출 파일을 생성합니다.")
    parser.add_argument('-c', '--checkpoint', required=True, help="사용할 모델의 .pth 체크포인트 파일 경로")
    parser.add_argument('-i', '--image_dir', default=cfg.TEST_DIRECTORY, help="예측을 수행할 테스트 이미지 폴더 경로")
    parser.add_argument('-o', '--output', default='submission.csv', help="결과를 저장할 CSV 파일명")
    # --- 이 부분을 추가 ---
    parser.add_argument(
        '--score_threshold', 
        type=float, 
        default=cfg.SCORE_THRESHOLD, # 기본 임계값 설정 (매우 낮은 값으로 시작)
        help="제출 파일에 포함될 예측의 최소 신뢰도 점수"
    )
    args = parser.parse_args()

    # --- 1. 매핑 정보 로드 ---
    # 훈련 데이터셋을 인스턴스화하여 라벨-ID 변환 맵을 가져옵니다.
    # 데이터를 실제로 사용하진 않으므로, 경로만 정확하면 됩니다.
    print("라벨 매핑 정보 로딩 중...")
    train_dataset = PillDataset(root=cfg.ROOT_DIRECTORY)
    label_to_cat_id = train_dataset.map_label_to_cat_id
    num_classes = train_dataset.get_num_classes()
    print("매핑 정보 로딩 완료.")

    # --- 2. 모델 로드 ---
    device = torch.device(cfg.DEVICE)
    # model = create_faster_rcnn_model(num_classes)
    model = create_model("faster_rcnn", num_classes).to(cfg.DEVICE)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # --- 3. 테스트 데이터 준비 ---
    image_paths = glob.glob(os.path.join(args.image_dir, '*.png'))
    test_dataset = PillTestDataset(image_paths, transforms=get_transform(train=False))
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    # --- 4. 추론 및 후처리 ---
    results = []
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="예측 수행 중")
        for images, filenames in progress_bar:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for i, output in enumerate(outputs):

                # 점수 임계값보다 높은 예측만 선택
                keep = output['scores'] > args.score_threshold
                
                # keep 마스크를 사용하여 필터링된 텐서들을 CPU로 이동
                boxes_cpu = output['boxes'][keep].cpu()
                labels_cpu = output['labels'][keep].cpu()
                scores_cpu = output['scores'][keep].cpu()

                for box, label, score in zip(boxes_cpu, labels_cpu, scores_cpu):
                    # --- 여기서 변환이 일어납니다! ---
                    original_category_id = label_to_cat_id[label.item()]

                    # bbox: [x_min, y_min, x_max, y_max] -> [x_min, y_min, width, height]
                    x, y, x_max, y_max = box.numpy()
                    w, h = x_max - x, y_max - y

                    results.append({
                        # TODO: int 필요한지 알아보기
                        'image_id': os.path.splitext(filenames[i])[0], # 예: '123.png' -> 123
                        'category_id': original_category_id,
                        'score': score.item(),
                        'bbox_x': x, 'bbox_y': y, 'bbox_w': w, 'bbox_h': h,
                    })

    # --- 5. 제출 파일 생성 ---
    # submission.csv 형식에 맞게 컬럼 순서 조정 필요
    if not results:
        print("경고: 예측된 결과가 없습니다. 헤더만 있는 빈 파일을 생성합니다.")
        submission_df = pd.DataFrame(columns=[
            'annotation_id', 'image_id', 'category_id', 'bbox_x', 
            'bbox_y', 'bbox_w', 'bbox_h', 'score'
        ])
    else:
        # 결과를 데이터프레임으로 변환
        submission_df = pd.DataFrame(results)

        # 1. 고유한 annotation_id 추가 (1부터 시작)
        submission_df['annotation_id'] = range(1, len(submission_df) + 1)

        # 2. 제출 형식에 맞게 컬럼 순서 재정렬
        column_order = [
            'annotation_id', 'image_id', 'category_id', 'bbox_x', 
            'bbox_y', 'bbox_w', 'bbox_h', 'score'
        ]
        submission_df = submission_df[column_order]


    submission_df.to_csv(args.output, index=False)
    print(f"제출 파일 '{args.output}' 생성이 완료되었습니다.")

if __name__ == '__main__':
    main()