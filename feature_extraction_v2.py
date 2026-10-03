import os
import glob
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import random
import argparse

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_i3d_model(num_classes=2, weight_path=None):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
    model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, num_classes)
    if weight_path and os.path.exists(weight_path):
        print(f"Loading weights from {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    return model

class I3DFeatureExtractor(nn.Module):
    def __init__(self, trained_model, fc_weights_path="fc_weights.pth"):
        super().__init__()
        self.backbone = nn.Sequential(*trained_model.blocks[:-1])
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(2048, 1024)
        
        # Load or Save fc weights to guarantee exactly the same projection space forever
        if os.path.exists(fc_weights_path):
            print(f"Loading FC weights from {fc_weights_path}")
            self.fc.load_state_dict(torch.load(fc_weights_path, map_location='cpu'))
        else:
            print(f"Initializing NEW FC weights and saving to {fc_weights_path}")
            torch.save(self.fc.state_dict(), fc_weights_path)

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x

def load_video_frames(video_path, resize=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def split_clips(frames, clip_len=16, stride=8, pad_last=True):
    clips = []
    i = 0
    while i + clip_len <= len(frames):
        clips.append(frames[i:i + clip_len])
        i += stride

    if pad_last and i < len(frames):
        last_clip = frames[-clip_len:]
        if len(last_clip) < clip_len:
            pad_frame = np.zeros_like(frames[0]) if len(frames) > 0 else np.zeros((224, 224, 3), dtype=np.uint8)
            while len(last_clip) < clip_len:
                last_clip.append(pad_frame)
        clips.append(last_clip)
    return clips

def extract_features_from_clips(clips, feature_model, device):
    features = []
    for clip in clips:
        clip_np = np.stack(clip).astype(np.float32) / 255.0
        clip_np = clip_np.transpose(3, 0, 1, 2)
        clip_tensor = torch.tensor(clip_np).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = feature_model(clip_tensor)
        features.append(feat.cpu().squeeze(0))
    return torch.stack(features)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_flag", type=int, default=0, help="0: No Finetuning (ft0), 1: Finetuning (ft1)")
    parser.add_argument("--stride", type=int, default=16, help="Clip extraction stride (e.g. 8 or 16)")
    parser.add_argument("--target", type=str, default="train", help="'train' for Preprocessed_match_data, 'test' for mt_ videos")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 로드
    weight_path = "best_i3d_model_v4.pth" if args.ft_flag == 1 else None
    model = get_i3d_model(num_classes=2, weight_path=weight_path)
    
    # Feature Extractor 초기화 (fc_weights.pth 유지)
    feature_model = I3DFeatureExtractor(model, fc_weights_path="fc_weights.pth").to(device)
    feature_model.eval()
    
    finetune_matches = [f"Normal_{i:03d}" for i in range(1, 11)] + [f"Bot_{i:03d}" for i in range(1, 11)]
    
    if args.target == "train":
        dataset_root = r"e:\01_Project\01_FPS_Cheat_Detection\raw_data\Preprocessed_match_data"
        save_dir = f"cs2_feat_{args.stride}_ft{args.ft_flag}"
        class_names = ["Normal_match", "Bot_match"]
        
        for label in class_names:
            class_dir = os.path.join(dataset_root, label)
            video_paths = glob.glob(os.path.join(class_dir, '**/*.mp4'), recursive=True)
            
            out_class_dir = os.path.join(save_dir, label)
            os.makedirs(out_class_dir, exist_ok=True)
            
            for vpath in tqdm(video_paths, desc=f"Extracting {label} (Stride {args.stride}, FT {args.ft_flag})"):
                filename = os.path.basename(vpath)
                match_id = filename.split('-')[0]
                
                # 파인튜닝에 사용된 매치는 피처 추출에서 제외 (시간 절약)
                if match_id in finetune_matches:
                    continue
                    
                save_path = os.path.join(out_class_dir, os.path.splitext(filename)[0] + ".npy")
                if os.path.exists(save_path):
                    continue
                    
                frames = load_video_frames(vpath)
                clips = split_clips(frames, clip_len=16, stride=args.stride, pad_last=True)
                if not clips:
                    continue
                    
                feat = extract_features_from_clips(clips, feature_model, device)
                np.save(save_path, feat.cpu().numpy())
                
    elif args.target == "test":
        dataset_root = r"e:\01_Project\01_FPS_Cheat_Detection\raw_data"
        save_dir = f"cs2_test_feat_{args.stride}_ft{args.ft_flag}"
        os.makedirs(save_dir, exist_ok=True)
        
        # mt_rep_ver, mt_new_ver 파일들
        video_paths = glob.glob(os.path.join(dataset_root, 'mt_*.mp4'))
        video_paths.extend(glob.glob(r"E:\01_Project\01_FPS_Cheat_Detection_Experiments\synthetic_test_data\*.mp4"))
        
        for vpath in tqdm(video_paths, desc=f"Extracting Test Videos (Stride {args.stride}, FT {args.ft_flag})"):
            filename = os.path.basename(vpath)
            save_path = os.path.join(save_dir, os.path.splitext(filename)[0] + ".npy")
            if os.path.exists(save_path):
                continue
                
            frames = load_video_frames(vpath)
            clips = split_clips(frames, clip_len=16, stride=args.stride, pad_last=True)
            if not clips:
                continue
                
            feat = extract_features_from_clips(clips, feature_model, device)
            np.save(save_path, feat.cpu().numpy())
            
    print(f"========== Feature Extraction Done ({args.target}, Stride {args.stride}, ft{args.ft_flag}) ==========")

if __name__ == "__main__":
    main()
