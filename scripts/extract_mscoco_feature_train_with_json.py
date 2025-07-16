import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import os
import numpy as np
from datasets import MSCOCODatabase
import argparse
from tqdm import tqdm
import json  

import libs.autoencoder
from libs.clip import FrozenCLIPEmbedder
from libs.t5 import T5Embedder


def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='val')
    args = parser.parse_args()
    print(args)

    if args.split == "val":
        datas = MSCOCODatabase(root='/data1/common_datasets/COCO/coco/images/train2017_256',
                             annFile='/data1/common_datasets/COCO/coco/annotations/captions_train2017.json',
                             size=resolution)
        save_dir = 'train'
    else:
        raise NotImplementedError

    device = "cuda"
    os.makedirs(save_dir, exist_ok=True)
    
    # 생성할 JSONL 파일 경로 정의
    jsonl_file_path ='metadata.jsonl'

    # 모델 로드
    autoencoder = libs.autoencoder.get_model('./assets/stable-diffusion/autoencoder_kl.pth')
    autoencoder.to(device)

    llm = 'clip'
    if llm == 'clip':
        text_encoder = FrozenCLIPEmbedder()
        text_encoder.eval()
        text_encoder.to(device)
    elif llm == 't5':
        text_encoder = T5Embedder(device=device)
    else:
        raise NotImplementedError

    # torch.no_grad()와 파일 쓰기를 한번에 관리
    with torch.no_grad(), open(jsonl_file_path, 'w', encoding='utf-8') as f_jsonl:
        # tqdm에 전체 길이를 넣어주면 진행률과 남은 시간을 더 정확히 표시합니다.
        for idx, data in tqdm(enumerate(datas), total=len(datas), desc="Extracting features"):
            x, captions = data

            if len(x.shape) == 3:
                x = x[None, ...]
         
            x = torch.tensor(x, device=device, dtype=torch.float32)
            
            # 1. 이미지 인코딩 및 피처 저장
            moments = autoencoder(x, fn='encode_moments').squeeze(0)
            moments_cpu = moments.detach().cpu().numpy()
            
            image_feature_filename = f'{idx}.npy'
            image_feature_path = os.path.join(save_dir, image_feature_filename)
            np.save(image_feature_path, moments_cpu)

            # 2. 캡션 인코딩
            if llm == 'clip':
                latent, latent_and_others = text_encoder.encode(captions)
                token_embedding = latent_and_others['token_embedding']
                token_mask = latent_and_others['token_mask']
                token = latent_and_others['tokens']
            elif llm == 't5':
                latent, latent_and_others = text_encoder.get_text_embeddings(captions)
                token_embedding = latent_and_others['token_embedding'].to(torch.float32) * 10.0
                token_mask = latent_and_others['token_mask']
                token = latent_and_others['tokens']

            # 3. 각 캡션별로 피처 저장 및 JSONL 파일에 메타데이터 기록
            for i in range(len(captions)):
                # 캡션 피처 데이터 준비
                caption_feature_data = {'prompt': captions[i],
                                        'token_embedding': token_embedding[i].detach().cpu().numpy(), 
                                        'token_mask': token_mask[i].detach().cpu().numpy(),
                                        'token': token[i].detach().cpu().numpy()}
                
                # 캡션 피처 저장
                caption_feature_filename = f'{idx}_{i}.npy'
                caption_feature_path = os.path.join(save_dir, caption_feature_filename)
                np.save(caption_feature_path, caption_feature_data)

                # JSONL 파일에 쓸 메타데이터 생성
                # 파일 경로는 전체가 아닌 상대 경로(파일명)만 저장하여 이식성을 높입니다.
                metadata_entry = {
                    # "image_feature": image_feature_filename,
                    # "caption_feature": caption_feature_filename
                    "img": f'{idx}.jpg',  # 이미지 파일명
                    "prompt": captions[i]
                }
                
                # JSON 형태로 변환하여 파일에 한 줄 쓰기
                f_jsonl.write(json.dumps(metadata_entry) + '\n')

    print(f"\n✅ 피처 추출 완료. 데이터 저장 경로: '{save_dir}'")
    print(f"✅ 메타데이터 파일 생성 경로: '{jsonl_file_path}'")


if __name__ == '__main__':
    main()