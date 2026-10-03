import os
import glob
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import random
from sklearn.model_selection import GroupShuffleSplit

# 1. 시드 고정 함수 (재현성 확보)
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 2. 비디오 전처리 함수
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

    while len(frames) < num_frames:
        zero_frame = np.zeros_like(frames[0]) if len(frames) > 0 else np.zeros((*resize, 3), dtype=np.uint8)
        frames.append(zero_frame)

    frames = frames[:num_frames]
    frames = np.stack(frames).astype(np.float32) / 255.0
    frames = frames.transpose(3, 0, 1, 2) # (C, T, H, W)
    return torch.tensor(frames)

# 3. Custom Dataset 클래스
class VideoDataset(Dataset):
    def __init__(self, root_dir, class_names):
        self.samples = []
        self.match_ids = []
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        for label in class_names:
            video_paths = glob.glob(os.path.join(root_dir, label, '**/*.mp4'), recursive=True)
            for path in video_paths:
                self.samples.append((path, self.class_to_idx[label]))
                filename = os.path.basename(path)
                match_id = filename.split('-')[0]
                self.match_ids.append(match_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video_tensor = read_video(video_path)
        return video_tensor, label

# 4. 모델 로딩 및 헤드 수정
def get_i3d_model(num_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
    model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, num_classes)
    return model

# 5. Early Stopping
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
        elif (self.mode == 'min' and score >= self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

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
    return total_loss / len(dataloader), correct / total

# 6. 훈련 함수
def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-4, patience=7, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopping(patience=patience, mode='min')
    best_model_path = "best_i3d_model_v4.pth"

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

        stopper(val_loss)
        if stopper.early_stop:
            print("Early stopping triggered.")
            break

        if val_loss == stopper.best_score:
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved")

    return model

if __name__ == '__main__':
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Normal_match", "Bot_match"]
    
    # 누수를 막기 위해, 딱 20개 매치만 있는 finetuning_data 경로 사용!
    dataset_root = r"e:\01_Project\01_FPS_Cheat_Detection\raw_data\finetuning_data"
    
    full_dataset = VideoDataset(dataset_root, class_names)
    print(f"Loaded {len(full_dataset)} clips for fine-tuning.")
    
    # 매치(Match) 단위로 Train/Val 분리 (Fine-tuning 단계에서도 완벽한 Leakage 차단)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(np.zeros(len(full_dataset)), groups=full_dataset.match_ids))
    
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    print(f"Train clips: {len(train_dataset)}, Val clips: {len(val_dataset)}")
    
    model = get_i3d_model(num_classes=len(class_names))
    train_model(model, train_loader, val_loader, num_epochs=50, patience=7, device=device)
    print("========== I3D Finetuning Done! ==========")
