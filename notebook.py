from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from lab_utils.visualization import extract_feature_maps, plot_feature_maps_like_reference, plot_training_history, show_tensor_batch

def find_project_root() -> Path:
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / 'data').exists():
            return candidate
    return Path.cwd().resolve()
PROJECT_ROOT = find_project_root()
DATA_ROOT = PROJECT_ROOT / 'data'
METADATA_PATH = DATA_ROOT / 'metadata.csv'
ARTIFACT_DIR = PROJECT_ROOT / 'artifacts'
LABELS = ('cat', 'dog')
SPLITS = ('train', 'val', 'test')
SEED = 1234
EPOCHS = 20
NUMPY_PRED_PATH = ARTIFACT_DIR / 'lab3_pytorch_predictions.csv'

def seed_index(length: int, offset: int=0) -> int:
    if length <= 0:
        raise ValueError('Cannot choose an index from an empty collection.')
    return int((SEED + offset) % length)

def build_metadata_from_folders(data_root: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for label in LABELS:
            label_dir = data_root / split / label
            for path in sorted(label_dir.glob('*.jpg')) + sorted(label_dir.glob('*.png')):
                with Image.open(path) as image:
                    image = image.convert('RGB')
                    width, height = image.size
                rows.append({'filepath': str(path.relative_to(data_root)), 'label': label, 'split': split, 'width': width, 'height': height})
    return pd.DataFrame(rows)

def build_label_mapping(frame: pd.DataFrame) -> tuple[dict[str, int], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labelled = frame.copy()
    labelled['label_id'] = labelled['label'].map(label_to_index)
    train_df = labelled[labelled['split'] == 'train'].copy()
    val_df = labelled[labelled['split'] == 'val'].copy()
    test_df = labelled[labelled['split'] == 'test'].copy()
    return (label_to_index, labelled, train_df, val_df, test_df)

def image_to_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    img = img.resize((64, 64))
    tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
    tensor = tensor.permute(2, 0, 1)
    return tensor

class CatsDogsDataset(Dataset):

    def __init__(self, frame: pd.DataFrame, data_root: Path):
        self.frame = frame.reset_index(drop=True)
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_tensor = image_to_tensor(self.data_root / row['filepath'])
        label_tensor = torch.tensor(row['label_id'], dtype=torch.long)
        return (image_tensor, label_tensor)
BATCH_SIZE = 32

def build_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, data_root: Path, batch_size: int=32, seed: int=SEED, dataset_cls: type[Dataset]=CatsDogsDataset) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = dataset_cls(train_df, data_root)
    val_dataset = dataset_cls(val_df, data_root)
    test_dataset = dataset_cls(test_df, data_root)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return (train_loader, val_loader, test_loader)

def inspect_first_batch(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    if loader is None:
        raise ValueError('Complete Question 3 before inspecting a batch.')
    batch_images, batch_labels = next(iter(loader))
    print('Image batch:', batch_images.shape, batch_images.dtype)
    print('Label batch:', batch_labels.shape, batch_labels.dtype)
    assert batch_images.ndim == 4, 'Batches of images should have shape (B, C, H, W).'
    assert batch_images.shape[1] == 3, 'Color images should have 3 channels.'
    assert batch_labels.dtype == torch.long, 'Labels should be torch.long class indices.'
    return (batch_images, batch_labels)

class CatsDogsSimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 16 * 16, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        return self.classifier(x)

def setup_training(model: nn.Module, device: torch.device | None=None, learning_rate: float=0.001) -> tuple[torch.device, nn.Module, nn.Module, torch.optim.Optimizer]:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return (device, model, criterion, optimizer)

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        preds = logits.argmax(dim=1)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_examples += batch_size
    average_loss = total_loss / total_examples
    average_accuracy = total_correct / total_examples
    return (average_loss, average_accuracy)

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_examples += batch_size
    average_loss = total_loss / total_examples
    average_accuracy = total_correct / total_examples
    return (average_loss, average_accuracy)

def run_training_experiment(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, epochs: int=5, plot: bool=True) -> tuple[list[dict[str, float]], float, float, float | None]:
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        record = {'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc}
        history.append(record)
        print(f'Epoch {epoch}/{epochs}  |  train loss={train_loss:.4f}  acc={train_acc:.3f}  |  val   loss={val_loss:.4f}  acc={val_acc:.3f}')
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'\nTest: loss={test_loss:.4f}, acc={test_acc:.3f}')
    if plot:
        epochs_list = [h['epoch'] for h in history]
        train_accs = [h['train_acc'] for h in history]
        val_accs = [h['val_acc'] for h in history]
        train_losses = [h['train_loss'] for h in history]
        val_losses = [h['val_loss'] for h in history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs_list, train_accs, label='Train acc')
        ax1.plot(epochs_list, val_accs, label='Val acc')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy over epochs')
        ax1.legend()
        ax2.plot(epochs_list, train_losses, label='Train loss')
        ax2.plot(epochs_list, val_losses, label='Val loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss over epochs')
        ax2.legend()
        plt.tight_layout()
        plt.show()
    return (history, test_loss, test_acc)
