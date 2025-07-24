# data/dataset.py
import torch
import os
import glob
import json
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict

# =================================================================================
# 1. 데이터셋 클래스 정의 (PillDataset)
# =================================================================================
class PillDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, min_box_size=10):  # min_box_size 인자 추가
        self.root = root
        self.transforms = transforms
        self.min_box_size = min_box_size # 추가

        # --- 최종 데이터를 담을 인스턴스 변수 초기화 ---
        self.images = []
        self.annotations = defaultdict(list)
        self.id_to_cat = {}
        self.map_cat_id_to_label = {}
        self.map_label_to_cat_id = {}

        # --- 모든 전처리를 수행하는 단일 메소드 호출 ---
        self._load_data()

        print(f"데이터셋 준비 완료: {len(self.images)}개 이미지, {len(self.map_cat_id_to_label)}개 클래스")


    def _load_data(self):
        """
        데이터를 로드하고 전처리하는 모든 과정을 통합하여 관리합니다.
        단 한 번의 파일 순회로 효율성을 극대화합니다.
        """
        print("데이터 로딩 및 전처리 시작...")
        #glob을 사용하여 재귀적으로 train_annotations 폴더 내의 모든 JSON 파일을 찾고 annotation_paths에 저장합니다.
        annotation_paths = glob.glob(os.path.join(self.root, 'train_annotations', '**', '*.json'), recursive=True)

        # 1. 임시 변수: 모든 정보를 한 번의 루프로 수집
        # 최종적으로 get_item에서 참조할 self.images와 self.annotations를 만들기 위한 임시변수들 입니다.
        raw_images = {}
        raw_annotations = defaultdict(list)
        present_category_ids = set() 

        #annotation_paths 안의 모든 JSON 파일을 순회하며 임시변수들에 정보를 수집합니다.
        for ann_path in tqdm(annotation_paths, desc="어노테이션 파일 처리 중"):
            # JSON 파일을 읽어들입니다.
            with open(ann_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 이미지 정보 수집
            for img in data.get('images', []):
                if img['id'] not in raw_images:
                    raw_images[img['id']] = img

            # 카테고리 이름 정보 수집
            for cat in data.get('categories', []):
                if cat['id'] not in self.id_to_cat:
                    self.id_to_cat[cat['id']] = cat['name']

            # 유효한 어노테이션 및 실제 사용된 카테고리 ID 수집
            for ann in data.get('annotations', []):
                bbox = ann.get('bbox', [])
                # bbox가 [x, y, w, h] 형식인지 확인하고 유효한 크기인지 검사
                if len(bbox) == 4 and bbox[2] > self.min_box_size and bbox[3] > self.min_box_size:
                    raw_annotations[ann['image_id']].append(ann)
                    present_category_ids.add(ann['category_id'])

        # 2. 클래스 매핑 생성
        sorted_ids = sorted(list(present_category_ids))
        # category_id와 모델 학습에 사용될 label ID를 매핑합니다.
        self.map_cat_id_to_label = {cat_id: i + 1 for i, cat_id in enumerate(sorted_ids)}
        # 위의 매핑을 역으로 생성하여 label ID로 category_id를 찾을 수 있도록 합니다.
        self.map_label_to_cat_id = {v: k for k, v in self.map_cat_id_to_label.items()}

        # 3. 최종 어노테이션 생성 (클래스 ID 변환): raw_annotations에서 실제로 존재하는 이미지 ID에 대해서만 self.annotations에 어노테이션을 추가합니다.
        for img_id, anns in raw_annotations.items():
            if raw_images.get(img_id):
                for ann in anns:
                    original_cat_id = ann['category_id']
                    if original_cat_id in self.map_cat_id_to_label:
                         # Albumentations는 bbox와 label을 분리하여 처리하므로,
                         # 여기서 category_id를 매핑된 라벨로 변환하는 대신
                         # __getitem__에서 변환하도록 합니다.
                         # ann['category_id'] = self.map_cat_id_to_label[original_cat_id]
                         self.annotations[img_id].append(ann)

        # 4. 최종 이미지 목록 생성 (유효한 어노테이션이 있는 이미지만)
        self.images = [img for img_id, img in raw_images.items() if img_id in self.annotations]
        self.images.sort(key=lambda x: x['file_name']) # 파일명 순으로 정렬하여 일관성 유지

    #index를 받아 해당 이미지(image)와 어노테이션 및 정답(target)을 반환합니다.
    def __getitem__(self, idx):
        # self.images에서 idx에 해당하는 이미지 정보를 가져옵니다.
        img_info = self.images[idx]
        # 이미지 경로를 생성합니다.
        img_path = os.path.join(self.root, 'train_images', img_info['file_name'])

        # image를 numpy 배열로 바로 읽습니다. (Albumentations 입력 형식)
        image = np.array(Image.open(img_path).convert("RGB"))

        #self.annotations에서 해당 이미지 ID에 대한 어노테이션을 가져옵니다.
        anns = self.annotations[img_info['id']]

        # for문을 통해 anns에서 bbox와 label을 추출하고 boxes와 labels 리스트를 만듭니다.
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            # Albumentations는 'pascal_voc' 형식을 사용할 때 [x_min, y_min, x_max, y_max]를 기대합니다.
            x1, y1, x2, y2 = x, y, x + w, y + h
            boxes.append([x1, y1, x2, y2])
            # 여기서 원본 category_id를 Albumentations의 label_fields에 전달합니다.
            # 변환 후 __getitem__ 끝에서 PyTorch에서 사용할 라벨로 매핑합니다.
            labels.append(ann['category_id']) # 원본 category_id 사용

        # Albumentations는 딕셔너리 형태로 입력 (image=..., boxes=..., labels=...)을 받습니다.
        # boxes와 labels는 리스트 형태여야 합니다.
        albumentations_input = {
            'image': image,
            'boxes': boxes,
            'labels': labels # 원본 category_id 리스트
        }

        #transforms가 None이 아니라면 image, boxes, labels에 모두 변환을 적용합니다. (ex) 이미지 회전 시 박스도 회전시킴)
        if self.transforms:
            # Albumentations 변환 적용
            transformed = self.transforms(**albumentations_input)
            image = transformed['image']
            boxes = transformed['boxes'] # 변환된 박스 (Albumentations 출력 형식)
            labels = transformed['labels'] # 변환된 라벨 (원본 category_id 유지)

        # 변환 후의 박스와 라벨을 PyTorch 텐서 형식으로 변환하고,
        # 라벨을 내부 매핑된 라벨 ID로 변환합니다.
        # Albumentations가 빈 boxes 리스트를 반환할 수 있으므로 확인 필요
        target_boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)

        # Albumentations에서 반환된 labels는 원본 category_id 리스트입니다.
        # 이를 내부적으로 사용하는 라벨 ID로 매핑합니다.
        # labels가 빈 리스트일 수 있으므로 확인 필요
        target_labels = torch.as_tensor([self.map_cat_id_to_label[cat_id] for cat_id in labels], dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        #target은 boxes와 labels 정보를 딕셔너리 형태로 포함합니다.
        target = {
            'boxes': target_boxes,
            'labels': target_labels
        }

        #image와 target(정답)을 반환합니다.
        return image, target

    #dataset 길이를 반환합니다.
    def __len__(self):
        return len(self.images)

    #num_classes(배경포함 74가지)를 반환합니다.
    def get_num_classes(self):
        return len(self.map_cat_id_to_label) + 1  # 배경 클래스 포함