import os
import glob
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# 1. 비디오 전처리 함수
def read_video(video_path, num_frames=16, resize=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)

    frames = []
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if len(frames) >= num_frames:
            break
    cap.release()

    # 부족하면 zero-padding
    while len(frames) < num_frames:
        zero_frame = np.zeros_like(frames[0])
        frames.append(zero_frame)

    # 넘치면 자르기
    frames = frames[:num_frames]

    # (T, H, W, C) → (C, T, H, W)
    frames = np.stack(frames).astype(np.float32) / 255.0
    frames = frames.transpose(3, 0, 1, 2)
    return torch.tensor(frames)


# 2. Custom Dataset 클래스
class VideoDataset(Dataset):
    def __init__(self, root_dir, class_names):
        self.samples = []
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        for label in class_names:
            video_paths = glob.glob(os.path.join(root_dir, label, '**/*.mp4'))
            for path in video_paths:
                self.samples.append((path, self.class_to_idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video_tensor = read_video(video_path)  # (C, T, H, W)
        return video_tensor, label

# 3. 모델 로딩 및 헤드 수정
def get_i3d_model(num_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
    model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, num_classes)  # Classifier 수정
    return model

# 4. 훈련 함수
def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4, patience=5):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopping(patience=patience, mode='min')  # 기준: validation loss
    best_model_path = "best_i3d_model_v3.pth"

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping
        stopper(val_loss)
        if stopper.early_stop:
            print("⏹️ Early stopping triggered.")
            break

        # 모델 저장
        if val_loss == stopper.best_score:
            torch.save(model.state_dict(), best_model_path)
            print("✅ Best model saved")

    return model

# 5. Feature Extractor 만들기
class I3DFeatureExtractor(nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        # classifier 레이어 제외한 I3D backbone 사용
        self.backbone = nn.Sequential(*trained_model.blocks[:-1])
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # (B, 2048, 1, 1, 1)
        self.fc = nn.Linear(2048, 1024)              # 차원 축소

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)     # (B, 2048, T', H', W')
            x = self.pool(x)         # (B, 2048, 1, 1, 1)
            x = x.view(x.size(0), -1)  # (B, 2048)
            x = self.fc(x)           # (B, 1024)
        return x

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_loss = total_loss / len(dataloader)
    val_acc = correct / total
    return val_loss, val_acc

class EarlyStopping:
    def __init__(self, patience=5, mode='min'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'min' and score >= self.best_score) or (self.mode == 'max' and score <= self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

from sklearn.model_selection import StratifiedShuffleSplit

def stratified_split(dataset, val_ratio=0.2, random_seed=42):
    targets = [label for _, label in dataset.samples]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_seed)
    indices = list(range(len(dataset)))
    train_idx, val_idx = next(splitter.split(indices, targets))
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    return train_subset, val_subset

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# 1. 비디오 프레임 불러오기
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
    return frames  # List of np.array(H, W, C)

# 2. 슬라이딩 클립 분할 (with overlap & padding)
def split_clips(frames, clip_len=16, stride=8, pad_last=True):
    clips = []
    i = 0
    while i + clip_len <= len(frames):
        clips.append(frames[i:i + clip_len])
        i += stride

    if pad_last and i < len(frames):  # 마지막 클립 패딩
        last_clip = frames[-clip_len:]
        if len(last_clip) < clip_len:
            pad_frame = np.zeros_like(frames[0])
            while len(last_clip) < clip_len:
                last_clip.append(pad_frame)
        clips.append(last_clip)

    return clips  # List of [clip_len x frame]

# 3. 클립별 특성 추출
def extract_features_from_clips(clips, feature_model, device):
    features = []
    for clip in clips:
        clip_np = np.stack(clip).astype(np.float32) / 255.0  # (T, H, W, C)
        clip_np = clip_np.transpose(3, 0, 1, 2)  # (C, T, H, W)
        clip_tensor = torch.tensor(clip_np).unsqueeze(0).to(device)  # (1, C, T, H, W)
        with torch.no_grad():
            feat = feature_model(clip_tensor)  # (1, 1024)
        features.append(feat.cpu().squeeze(0))  # (1024,)
    return torch.stack(features)  # (N, 1024)

def process_dataset_and_save_per_video(dataset, class_names, feature_model, device, save_dir="./features_npy_per_video", clip_len=16, stride=8, pad_last=True):
    feature_model.eval().to(device)
    os.makedirs(save_dir, exist_ok=True)

    for i in tqdm(range(len(dataset)), desc="Processing dataset"):
        video_path, label = dataset.samples[i]
        frames = load_video_frames(video_path)
        clips = split_clips(frames, clip_len=clip_len, stride=stride, pad_last=pad_last)
        if len(clips) == 0:
            continue
        feat = extract_features_from_clips(clips, feature_model, device)  # (N, 1024)

        # 클래스별 폴더 생성
        class_name = class_names[label]
        class_folder = os.path.join(save_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)

        # 원본 비디오 파일명으로 저장 (확장자 제외)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        save_path = os.path.join(class_folder, base_name + ".npy")

        np.save(save_path, feat.cpu().numpy())
        print(f"✅ Saved {feat.shape} → {save_path}")


if __name__ == '__main__':
    # 기본 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Normal_match", "Bot_match"]
    dataset_root = "./raw_data/Preprocessed_match_data"

    # 전체 데이터셋 로딩
    full_dataset = VideoDataset(dataset_root, class_names)

    # Stratified split
    # train_dataset, val_dataset = stratified_split(full_dataset, val_ratio=0.2)
    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # 모델 준비 및 학습
    model = get_i3d_model(num_classes=len(class_names))
    #model = train_model(model, train_loader, val_loader, num_epochs=50, patience=7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model = get_i3d_model(num_classes=2)
    #model.load_state_dict(torch.load('./best_i3d_model_v3.pth', map_location='cuda:0'))
    feature_model = I3DFeatureExtractor(model).to(device)
    feature_model.eval()
    full_dataset = VideoDataset("./raw_data/Preprocessed_match_data", class_names)
    for ft_flag in [0, 1]:
        for i in [1, 2, 4]:
            process_dataset_and_save_per_video(
                dataset=full_dataset,
                class_names=class_names,
                feature_model=feature_model,
                device=device,
                save_dir=f"./cs2_feat_{i}_ft{ft_flag}",
                clip_len=16,
                stride=i,
                pad_last=True
            )

        model.load_state_dict(torch.load('./best_i3d_model_v3.pth', map_location='cuda:0'))
        feature_model = I3DFeatureExtractor(model).to(device)
        feature_model.eval()
    print("========== I3D Finetuning and Extraction Done! ==========")