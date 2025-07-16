import os
import shutil
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from tqdm import tqdm
import json
from torchvision.datasets import CocoCaptions
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# --- 필요한 라이브러리 임포트 ---
import libs.autoencoder
from libs.clip import FrozenCLIPEmbedder

def main(batch_size=32):
    # --- 1. 경로 및 설정 ---
    # COCO 데이터셋 경로를 설정해주세요.
    coco_root_dir = '/data1/common_datasets/COCO/coco/images/train2017_256'
    coco_ann_file = '/data1/common_datasets/COCO/coco/annotations/captions_train2017.json'
    
    # 피처와 메타데이터를 저장할 디렉토리
    save_dir = 'mscoco_features_256'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 기존 폴더가 있다면 삭제하고 새로 생성
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # --- 2. 모델 로드 ---
    print("Loading models...")
    # Autoencoder 모델
    autoencoder = libs.autoencoder.get_model('./assets/stable-diffusion/autoencoder_kl.pth')
    autoencoder.to(device)
    autoencoder.eval()

    # CLIP 모델
    clip = FrozenCLIPEmbedder()
    clip.to(device)
    clip.eval()
    print("Models loaded successfully.")

    # --- 3. 데이터셋 및 데이터로더 준비 ---
    # 이미지 전처리: 256x256으로 리사이즈 및 크롭 후 [-1, 1] 범위로 정규화
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),  # 0-255 -> [0.0, 1.0]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [0.0, 1.0] -> [-1.0, 1.0]
    ])

    print("Loading MSCOCO dataset...")
    # CocoCaptions는 (이미지, 캡션_리스트) 형태의 튜플을 반환합니다.
    coco_dataset = CocoCaptions(root=coco_root_dir, annFile=coco_ann_file, transform=transform)

    # (이미지, 캡션) 쌍으로 데이터를 풀어주기 위한 전처리
    flat_data = []
    print("Preparing image-caption pairs...")
    for i in tqdm(range(len(coco_dataset)), desc="Flattening data"):
        img_tensor, captions_list = coco_dataset[i]
        img_id = coco_dataset.ids[i]  # COCO 고유 이미지 ID
        for cap_idx, caption in enumerate(captions_list):
            flat_data.append({
                "image": img_tensor,
                "caption": caption,
                "filename": f"{img_id}_{cap_idx}"  # 고유 파일명 생성 (이미지ID_캡션인덱스)
            })

    # 커스텀 collate 함수: 딕셔너리 리스트를 배치로 묶어줌
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        captions = [item['caption'] for item in batch]
        filenames = [item['filename'] for item in batch]
        return images, captions, filenames

    # 데이터로더 생성
    data_loader = DataLoader(flat_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # --- 4. 피처 추출 및 저장 ---
    print(f"\nStarting feature extraction for {len(flat_data)} pairs...")
    jsonl_path = 'metadata.jsonl'
    saved_count = 0

    with torch.no_grad(), open(jsonl_path, 'w', encoding='utf-8') as f_jsonl:
        for batch_images, batch_captions, batch_filenames in tqdm(data_loader, desc="Processing batches"):
            batch_images = batch_images.to(device)

            # 이미지 피처 추출
            moments_256 = autoencoder(batch_images, fn='encode_moments').detach().cpu()

            # 텍스트 피처 추출
            _, latent_and_others_clip = clip.encode(batch_captions)
            token_embedding_clip = latent_and_others_clip['token_embedding'].detach().cpu()
            token_mask_clip = latent_and_others_clip['token_mask'].detach().cpu()
            token_clip = latent_and_others_clip['tokens'].detach().cpu()

            # 배치 내 각 아이템에 대해 파일 저장 및 jsonl 기록
            for i in range(len(batch_filenames)):
                feature_filename = f"{batch_filenames[i]}.npy"
                target_path = os.path.join(save_dir, feature_filename)
                
                # 저장할 피처 데이터
                data_to_save = {
                    'image_latent_256': moments_256[i].numpy(),
                    'token_embedding_clip': token_embedding_clip[i].numpy(),
                    'token_mask_clip': token_mask_clip[i].numpy(),
                    'token_clip': token_clip[i].numpy()
                }
                np.save(target_path, data_to_save)

                # JSONL 메타데이터 기록
                json_entry = {
                    "img": f"{batch_filenames[i].split('_')[0]}.jpg",  # 이미지 파일명 (이미지 ID)
                    "prompt": batch_captions[i]
                }
                f_jsonl.write(json.dumps(json_entry) + '\n')
                saved_count += 1

    print("-" * 30)
    print(f"✅ Successfully saved {saved_count} feature files in '{save_dir}'.")
    print(f"✅ Metadata file created at '{jsonl_path}'.")

if __name__ == '__main__':
    main()