# 모델 .pth파일이 저장된 경로
num_classes = 73 + 1

checkpoint = "/content/drive/MyDrive/코드잇/초급 프로젝트 /checkpoint/checkpoints_0723_Augmentation/v2/get_augmented_transform1, mAP: 0.8781/EPOCH(14)-LOSS(0.1499).pth"

# --- 모델 로드 ---
model = create_faster_rcnn_model(num_classes)
try:
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
except FileNotFoundError:
    print(f"오류: 체크포인트 파일을 찾을 수 없습니다 - {checkpoint}")
model.to(DEVICE)

# 이미지 찾기
image_dir = "/content/drive/MyDrive/코드잇/초급 프로젝트 /data/test_images"
# image_paths = sorted(glob.glob(os.path.join(ROOT_DIRECTORY, 'test_images', '*.png')))

image_paths = sorted(glob.glob(os.path.join(ROOT_DIRECTORY, 'test_images', '*.png')), key=numerical_sort_key)

print(image_paths)
print(f"\n이미지 수: {len(image_paths)}")

from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import os
import pandas as pd

# 이미지 변환 정의 (훈련 시 사용한 것과 동일하게)
transform = get_transform(train=False)

predictions = []

print("알약 탐지 시작...")

# image_paths는 이미 마지막 셀에서 numerical_sort_key를 사용하여 정렬되었습니다.
for img_path in tqdm(image_paths, desc="이미지 탐지 중"):
    # 이미지 로드
    try:
        image = Image.open(img_path).convert('RGB')
    except FileNotFoundError:
        print(f"경고: 이미지를 찾을 수 없습니다 - {img_path}. 건너뜁니다.")
        continue

    # 이미지를 텐서로 변환 및 장치 이동
    # torchvision v2 transform은 이미지와 타겟을 함께 받지만, 예측 시에는 타겟이 없습니다.
    # 따라서 이미지에만 transform을 적용해야 합니다.
    # 여기서는 get_transform이 ToImage와 ToDtype만 포함하므로 직접 적용합니다.
    # 만약 RandomHorizontalFlip 같은 증강이 포함된 transform이라면 예측 시에는 train=False로 호출해야 합니다.
    image_tensor = T.ToImage()(image)
    image_tensor = T.ToDtype(torch.float32, scale=True)(image_tensor)
    image_tensor = image_tensor.to(DEVICE)

    # 배치 차원 추가 (모델 입력 형식에 맞게)
    image_tensor = image_tensor.unsqueeze(0)


    # 예측 실행
    model.eval() # 모델을 평가 모드로 설정
    with torch.no_grad():
        prediction = model(image_tensor)

    # 결과 후처리 (예: 특정 임계값 이하의 낮은 점수 제거 등)
    # 이 부분은 모델 출력 형식에 따라 다를 수 있습니다.
    # Faster R-CNN의 경우, prediction은 예측 결과 딕셔너리 리스트입니다.
    # 각 딕셔너리는 'boxes', 'labels', 'scores' 키를 가집니다.
    # 여기서는 첫 번째 이미지의 결과만 사용합니다.
    output = prediction[0]

    # 점수 임계값 적용 (예시: 0.05 - 대회 기준에 맞게 조정)
    keep = output['scores'] > SCORE_THRESHOLD

    boxes = output['boxes'][keep]
    labels = output['labels'][keep]
    scores = output['scores'][keep]

    # predictions 리스트에 추가할 때 필요한 정보만 저장
    predictions.append({
        'image_id': int(os.path.splitext(os.path.basename(img_path))[0]), # 파일 이름(숫자) 추출하여 int로 저장
        'boxes': boxes.cpu().numpy(),
        'labels': labels.cpu().numpy(),
        'scores': scores.cpu().numpy()
    })

print("알약 탐지 완료.")

# =================================================================================
# 탐지 결과를 CSV 파일로 변환 및 저장
# =================================================================================

csv_data = []
annotation_id_counter = 1 # annotation_id는 1부터 시작하여 순차적으로 증가

for pred in predictions:
    image_id = pred['image_id']
    boxes = pred['boxes']
    labels = pred['labels']
    scores = pred['scores']

    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i]
        category_id = label_to_map_cat_id[label]
        score = scores[i]

        # bbox 좌표 변환 (x_min, y_min, x_max, y_max) -> (x, y, w, h)
        bbox_x = box[0]
        bbox_y = box[1]
        bbox_w = box[2] - box[0]
        bbox_h = box[3] - box[1]

        # CSV 데이터 형식에 맞게 리스트에 추가
        csv_data.append([
            annotation_id_counter,
            image_id,
            category_id,
            bbox_x,
            bbox_y,
            bbox_w,
            bbox_h,
            score
        ])
        annotation_id_counter += 1

# DataFrame 생성
csv_df = pd.DataFrame(csv_data, columns=[
    'annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score'
])

# CSV 파일로 저장 (예: submission.csv)
output_csv_path = os.path.join(CHECKPOINT_DIR, 'submission.csv') # 체크포인트 디렉토리에 저장
csv_df.to_csv(output_csv_path, index=False)

print(f"\n탐지 결과가 {output_csv_path} 파일로 저장되었습니다.")