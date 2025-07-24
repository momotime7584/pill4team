# data/dataset.py
import torch
import os
import glob
import json
from PIL import Image
from tqdm.auto import tqdm

# =================================================================================
# 1. 데이터셋 클래스 정의 (PillDataset)
# =================================================================================
class PillDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.annotation_paths = sorted(glob.glob(os.path.join(self.root, 'train_annotations', '**', '*.json'), recursive=True))

        self.categories = self._get_all_categories()
        self.cat_to_id = {cat['name']: cat['id'] for cat in self.categories}
        self.id_to_cat = {cat['id']: cat['name'] for cat in self.categories}

        self.class_ids = sorted(self.cat_to_id.values())
        self.map_cat_id_to_label = {cat_id: i + 1 for i, cat_id in enumerate(self.class_ids)}
        self.map_label_to_cat_id = {v: k for k, v in self.map_cat_id_to_label.items()}

        self.image_paths = sorted(glob.glob(os.path.join(self.root, 'train_images', '*.png')))
        self.image_to_labels = self._get_all_labels()

        print(f"총 {len(self.annotation_paths)}개의 annotation 파일을 찾았습니다.")
        print(f"총 {len(self.class_ids)}개의 고유한 클래스를 발견했습니다.")

    def _get_all_labels(self):
       image_to_labels = {}
       for ann_path in tqdm(self.annotation_paths, desc="라벨 정보 로딩 중"):

            with open(ann_path, 'r') as f:
                data = json.load(f)

            image_info = data['images'][0]
            file_name = image_info['file_name']
            # img_path = os.path.join(self.root, 'train_images', file_name)

            ann = data['annotations'][0]
            # 유효한 어노테이션이 없는 경우 건너뛰기
            #if not annotations or not any(ann.get('bbox') for ann in annotations):
            #    return None

            # bbox 정보가 없거나 유효하지 않은 어노테이션은 건너뜁니다.
            if 'bbox' not in ann or not ann['bbox']:
                continue
            x_min, y_min, w, h = ann['bbox']
            # bbox가 비정상적인 경우 건너뜁니다.
            if w <= 0 or h <= 0:
                continue
            x_max, y_max = x_min + w, y_min + h
            box = torch.as_tensor([x_min, y_min, x_max, y_max], dtype=torch.int64)

            original_cat_id = ann['category_id']
            label = self.map_cat_id_to_label[original_cat_id]
            label = torch.as_tensor([label], dtype=torch.int64)

            # 유효한 박스가 하나도 없으면 이 데이터는 무시합니다.
            #if not box:
            #    return None

            #image_id = torch.tensor([image_info['file_name']])
            #area = torch.tensor((box[3] - box[1]) * (box[2] - box[0]))
            #iscrowd = torch.zeros((1), dtype=torch.int64)

            if file_name in image_to_labels:
                image_to_labels[file_name]['boxes'] = torch.cat([image_to_labels[file_name]['boxes'], box.unsqueeze(0)], dim=0)
                image_to_labels[file_name]['labels'] = torch.cat([image_to_labels[file_name]['labels'], label], dim=0)
                #image_to_labels[file_name]['image_id'] = torch.cat([image_to_labels[file_name]['image_id'], image_id], dim=0)
                #image_to_labels[file_name]['area'] = torch.cat([image_to_labels[file_name]['area'], area], dim=0)
                #image_to_labels[file_name]['iscrowd'] = torch.cat([image_to_labels[file_name]['iscrowd'], area], dim=0)
            else:
                image_to_labels[file_name] = {}
                image_to_labels[file_name]['boxes'] =  box.unsqueeze(0)
                image_to_labels[file_name]['labels'] =  label
                #image_to_labels[file_name]['image_id'] = image_id
                #image_to_labels[file_name]['area'] = area
                #image_to_labels[file_name]['iscrowd'] = iscrowd

       return image_to_labels


    def _get_all_categories(self):
        all_cats = {}
        # tqdm을 사용해 카테고리 로딩 진행 상황을 보여줍니다.
        for ann_path in tqdm(self.annotation_paths, desc="카테고리 정보 로딩 중"):
            with open(ann_path, 'r') as f:
                data = json.load(f)
                if 'categories' in data:
                    for cat in data['categories']:
                        if cat['id'] not in all_cats:
                            all_cats[cat['id']] = cat
        return list(all_cats.values())


    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)

        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Colab 환경에서는 파일을 못찾으면 다음으로 넘어가는게 중요
            return None


        annotation = self.image_to_labels[filename]

        target = {}
        target["boxes"] = annotation["boxes"]
        target["labels"] = annotation["labels"]
        #target["image_id"] = annotation["image_id"]
        #target["area"] = annotation["area"]
        #target["iscrowd"] = annotation["iscrowd"]

        if self.transforms:
            # torchvision v2 transform은 이미지와 타겟을 함께 받습니다.
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_paths)

    def get_num_classes(self):
        return len(self.class_ids) + 1  # 배경 클래스 포함