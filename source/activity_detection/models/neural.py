from __future__ import annotations

import copy
import os
import random
from typing import Dict, Optional

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..interfaces import Model
from ..optional import (
    TORCH_AVAILABLE,
    DataLoader,
    T,
    TorchDataset,
    nn,
    torch,
    tv_models,
)
from ..registry import register

if TORCH_AVAILABLE:
    # In-memory frame cache, so repeated epochs do not decode every video from
    # disk again.
    _FRAME_CACHE: Dict[str, list] = {}
    _FRAME_CACHE_MAX_ITEMS = 256

    def _compute_class_weights(y_enc: np.ndarray, n_classes: int) -> "torch.Tensor":
        """Inverse-frequency class weights for CrossEntropyLoss (the classes are imbalanced)."""
        counts = np.bincount(y_enc, minlength=n_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (n_classes * counts)
        return torch.tensor(weights, dtype=torch.float32)

    # CPU-bound video decoding") ─────────────────────────────────────────────
    # Video decoding/resizing in VideoClipDataset.__getitem__ runs on CPU no
    # matter how the GPU is used, so the fix is to overlap that CPU work with
    # GPU compute rather than to make the GPU itself do more. This matters
    # *more* on a modest GPU: a weak card finishes each batch's forward/backward
    # pass quickly, so it is proportionally more exposed to CPU decode stalls
    # than a powerful one would be.
    _NUM_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))

    def _make_loader(dataset, batch_size, shuffle, num_workers=None):
        """DataLoader factory used by every Group C model (CNN, CNN+LSTM,
        3D CNN) so train/val/predict loaders all get the same treatment:
        background workers decode/augment the next batch on CPU while the
        GPU is busy with the current one, pinned memory speeds up the CPU->GPU
        copy, and persistent_workers avoids re-spawning worker processes every
        epoch (video decoding worker start-up is not free)."""
        nw = _NUM_WORKERS if num_workers is None else num_workers
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None,
        )

    def _to_device(x, device):
        """.to() with non_blocking=True — only actually asynchronous when
        paired with pin_memory=True (see _make_loader), otherwise behaves
        like a normal blocking transfer."""
        return x.to(device, non_blocking=True)

    class _AmpHelper:
        """Automatic Mixed Precision wrapper. On a basic/low-VRAM GPU this is
        usually a bigger win than on a powerful one: fp16 activations roughly
        halve memory use, which both speeds up the conv/LSTM math and leaves
        headroom to raise batch_size instead of being stuck at 4-8. Silently
        becomes a no-op on CPU-only machines."""

        def __init__(self, device: "torch.device", enabled: Optional[bool] = None):
            self.enabled = (enabled if enabled is not None else torch.cuda.is_available())
            self._scaler = torch.cuda.amp.GradScaler(enabled=self.enabled)
            self._device_type = "cuda" if device.type == "cuda" else "cpu"

        def autocast(self):
            return torch.autocast(device_type=self._device_type, enabled=self.enabled)

        def backward(self, loss):
            self._scaler.scale(loss).backward()

        def step(self, optimizer):
            self._scaler.step(optimizer)
            self._scaler.update()

    class VideoClipDataset(TorchDataset):
        def __init__(self, video_paths, labels, clip_len=16, frame_size=(112, 112), sampling="uniform",
                     transform=None, train: bool = False, augment: bool = False):
            self.video_paths = video_paths
            self.labels = labels
            self.clip_len = clip_len
            self.frame_size = frame_size
            self.sampling = sampling
            self.train = train
            # Light, video-safe augmentation applied only to the training split.
            # Kept deliberately mild — flip / crop / brightness jitter only —
            # so per-frame transforms stay consistent and don't introduce
            # spurious motion artefacts within a clip.
            self.augment = augment and train
            if transform is not None:
                self.transform = transform
            elif self.augment:
                self.transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize((int(frame_size[0] * 1.15), int(frame_size[1] * 1.15))),
                    T.RandomCrop(frame_size),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ColorJitter(brightness=0.2, contrast=0.2),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                self.transform = T.Compose([
                    T.ToPILImage(), T.Resize(frame_size), T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        def __len__(self): return len(self.video_paths)

        def __getitem__(self, idx):
            path = self.video_paths[idx]
            frames = self._extract_frames(path)
            if len(frames) >= self.clip_len:
                if self.sampling == "uniform":
                    indices = np.linspace(0, len(frames) - 1, self.clip_len, dtype=int)
                else:
                    start = random.randint(0, len(frames) - self.clip_len)
                    indices = range(start, start + self.clip_len)
            else:
                indices = list(range(len(frames)))
                while len(indices) < self.clip_len:
                    indices.append(len(frames) - 1)
            # Apply the same random crop/flip decision across every frame in
            # the clip so augmentation doesn't introduce inter-frame jitter
            # that would look like spurious motion.
            if self.augment:
                seed = random.randint(0, 2**31 - 1)
                frames_out = []
                for i in indices[:self.clip_len]:
                    torch.manual_seed(seed)
                    random.seed(seed)
                    frames_out.append(self.transform(frames[i]))
                clip = torch.stack(frames_out)
            else:
                clip = torch.stack([self.transform(frames[i]) for i in indices[:self.clip_len]])
            return clip, torch.tensor(self.labels[idx])

        def _extract_frames(self, path):
            # Cache key includes frame_size since we now store *resized*
            # frames (see below) rather than raw full-resolution frames.
            cache_key = (path, self.frame_size)
            cached = _FRAME_CACHE.get(cache_key)
            if cached is not None:
                return cached
            cap = cv2.VideoCapture(str(path))
            frames = []
            # Resize target as (width, height) for cv2.resize, derived from
            # frame_size which is (H, W) elsewhere in this class.
            resize_wh = (self.frame_size[1], self.frame_size[0])
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Downscale immediately, before converting/appending. The
                # original full-resolution frames (e.g. 1920x956, ~5.25MB
                # each) were being kept in RAM and cached across up to 256
                # videos, which is what was blowing through available
                # memory (numpy/cv2/torch OOM errors) — nothing downstream
                # ever needs more than frame_size pixels per frame.
                frame = cv2.resize(frame, resize_wh, interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            if not frames:
                frames = [np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)]
            if len(_FRAME_CACHE) < _FRAME_CACHE_MAX_ITEMS:
                _FRAME_CACHE[cache_key] = frames
            return frames

    def _train_val_split_paths(X, y, val_size=0.15, seed=42):
        """Split off a validation set for early stopping. Falls back to a
        non-stratified split (or no split, if too few samples) gracefully,
        mirroring the stratification fallback used elsewhere in the pipeline."""
        y = np.asarray(y)
        n_classes = len(np.unique(y))
        if len(y) < max(4, n_classes * 2):
            # Too small to carve out a validation set — train on everything,
            # skip early stopping for this fold.
            return X, [], y, np.array([])
        try:
            return train_test_split(X, y, test_size=val_size, stratify=y, random_state=seed)
        except ValueError:
            return train_test_split(X, y, test_size=val_size, random_state=seed)

    @register("model", "cnn")
    class CNNModel(Model):
        """2D-CNN baseline.

        Averaging raw frames into one static image before the backbone would discard
        motion and ordering, so each frame is passed through the ResNet backbone individually
        and the resulting *feature vectors* are averaged ("late pooling"),
        which preserves far more of the per-frame visual information while
        keeping the model a lightweight 2D-CNN comparison baseline (the
        ordering-aware comparison is CNNLSTMModel / CNN3DModel).
        """

        def __init__(self, backbone="resnet18_2d", pretrained=False, lr=0.001, epochs=15, batch_size=8,
                     random_state=42, clip_len=16, frame_size=(112, 112),
                     val_size=0.15, patience=5, use_class_weights=True, augment=True):
            self.backbone = backbone
            self.pretrained = pretrained
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.random_state = random_state
            self.clip_len = clip_len
            self.frame_size = frame_size
            self.val_size = val_size
            self.patience = patience
            self.use_class_weights = use_class_weights
            self.augment = augment
            self._model = None
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._amp = _AmpHelper(self._device)
            self._le = LabelEncoder()
            self._classes = None

        def _build_backbone(self, n_classes):
            model = tv_models.resnet18(pretrained=self.pretrained)
            model.fc = nn.Linear(512, n_classes)
            return model.to(self._device)

        def _forward_pooled(self, clips):
            # clips: (B, T, C, H, W) -> per-frame backbone features averaged
            # over T, then classified. This is what replaces the old
            # `clips.mean(dim=1)` raw-pixel averaging.
            B, T, C, H, W = clips.shape
            feats = self._features(clips.view(B * T, C, H, W))
            feats = feats.view(B, T, -1).mean(dim=1)
            return self._classifier(feats)

        def fit(self, X, y):
            self._le.fit(y)
            y_enc = self._le.transform(y)
            self._classes = self._le.classes_
            n_classes = len(self._classes)
            backbone = self._build_backbone(n_classes)
            self._features = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten()).to(self._device)
            self._classifier = nn.Linear(512, n_classes).to(self._device)
            self._model = backbone  # kept for checkpointing compatibility

            X_tr, X_val, y_tr, y_val = _train_val_split_paths(X, y_enc, val_size=self.val_size, seed=self.random_state)
            params = list(self._features.parameters()) + list(self._classifier.parameters())
            optimizer = torch.optim.Adam(params, lr=self.lr)
            weight = _compute_class_weights(y_tr, n_classes).to(self._device) if self.use_class_weights else None
            criterion = nn.CrossEntropyLoss(weight=weight)

            train_ds = VideoClipDataset(X_tr, y_tr, clip_len=self.clip_len, frame_size=self.frame_size,
                                         train=True, augment=self.augment)
            train_loader = _make_loader(train_ds, self.batch_size, shuffle=True)
            val_loader = None
            if len(X_val):
                val_ds = VideoClipDataset(X_val, y_val, clip_len=self.clip_len, frame_size=self.frame_size, train=False)
                val_loader = _make_loader(val_ds, self.batch_size, shuffle=False)

            best_val_loss = float("inf")
            best_state = None
            epochs_no_improve = 0
            for epoch in range(self.epochs):
                self._features.train(); self._classifier.train()
                total_loss = 0
                for clips, labels in train_loader:
                    clips = _to_device(clips, self._device)
                    labels = _to_device(labels, self._device)
                    optimizer.zero_grad()
                    with self._amp.autocast():
                        loss = criterion(self._forward_pooled(clips), labels)
                    self._amp.backward(loss)
                    self._amp.step(optimizer)
                    total_loss += loss.item()
                msg = f"    Epoch {epoch+1}/{self.epochs}  train_loss={total_loss/len(train_loader):.4f}"

                if val_loader is not None:
                    val_loss = self._eval_loss(val_loader, criterion)
                    msg += f"  val_loss={val_loss:.4f}"
                    if val_loss < best_val_loss - 1e-4:
                        best_val_loss = val_loss
                        best_state = (copy.deepcopy(self._features.state_dict()),
                                      copy.deepcopy(self._classifier.state_dict()))
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    print(msg)
                    if epochs_no_improve >= self.patience:
                        print(f"    Early stopping at epoch {epoch+1} (patience={self.patience})")
                        break
                else:
                    print(msg)

            if best_state is not None:
                self._features.load_state_dict(best_state[0])
                self._classifier.load_state_dict(best_state[1])
            return self

        def _eval_loss(self, loader, criterion):
            self._features.eval(); self._classifier.eval()
            total, n = 0.0, 0
            with torch.no_grad():
                for clips, labels in loader:
                    clips = _to_device(clips, self._device)
                    labels = _to_device(labels, self._device)
                    with self._amp.autocast():
                        loss = criterion(self._forward_pooled(clips), labels)
                    total += loss.item() * len(labels)
                    n += len(labels)
            return total / max(n, 1)

        def predict(self, X):
            if self._model is None: raise ValueError("Model not fitted")
            self._features.eval(); self._classifier.eval()
            dataset = VideoClipDataset(X, [0]*len(X), clip_len=self.clip_len, frame_size=self.frame_size)
            loader = _make_loader(dataset, self.batch_size, shuffle=False)
            preds = []
            with torch.no_grad():
                for clips, _ in loader:
                    clips = _to_device(clips, self._device)
                    with self._amp.autocast():
                        out = self._forward_pooled(clips)
                    preds.extend(out.argmax(dim=1).cpu().numpy())
            return self._le.inverse_transform(np.array(preds))

        def predict_proba(self, X):
            if self._model is None: return None
            self._features.eval(); self._classifier.eval()
            dataset = VideoClipDataset(X, [0]*len(X), clip_len=self.clip_len, frame_size=self.frame_size)
            loader = _make_loader(dataset, self.batch_size, shuffle=False)
            probas = []
            with torch.no_grad():
                for clips, _ in loader:
                    clips = _to_device(clips, self._device)
                    with self._amp.autocast():
                        logits = self._forward_pooled(clips)
                    probas.extend(torch.softmax(logits, dim=1).float().cpu().numpy())
            return np.array(probas)

        @property
        def classes_(self): return self._classes

    @register("model", "cnn_lstm")
    class CNNLSTMModel(Model):
        def __init__(self, backbone="resnet18_2d", hidden_size=256, num_layers=1, lr=0.0005, epochs=20,
                     batch_size=8, random_state=42, clip_len=16, frame_size=(112, 112),
                     val_size=0.15, patience=5, use_class_weights=True, augment=True):
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.random_state = random_state
            self.clip_len = clip_len
            self.frame_size = frame_size
            self.val_size = val_size
            self.patience = patience
            self.use_class_weights = use_class_weights
            self.augment = augment
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._amp = _AmpHelper(self._device)
            self._le = LabelEncoder()
            self._classes = None
            self._cnn = None
            self._lstm = None
            self._classifier = None

        def _build(self, n_classes):
            cnn = tv_models.resnet18(pretrained=False)
            cnn.fc = nn.Identity()
            self._cnn = cnn.to(self._device)
            self._lstm = nn.LSTM(512, self.hidden_size, self.num_layers, batch_first=True).to(self._device)
            self._classifier = nn.Linear(self.hidden_size, n_classes).to(self._device)

        def _forward(self, clips):
            B, T, C, H, W = clips.shape
            feats = self._cnn(clips.view(B * T, C, H, W)).view(B, T, -1)
            out, _ = self._lstm(feats)
            return self._classifier(out[:, -1, :])

        def fit(self, X, y):
            self._le.fit(y)
            y_enc = self._le.transform(y)
            self._classes = self._le.classes_
            self._build(len(self._classes))
            X_tr, X_val, y_tr, y_val = _train_val_split_paths(X, y_enc, val_size=self.val_size, seed=self.random_state)
            params = list(self._cnn.parameters()) + list(self._lstm.parameters()) + list(self._classifier.parameters())
            optimizer = torch.optim.Adam(params, lr=self.lr)
            weight = _compute_class_weights(y_tr, len(self._classes)).to(self._device) if self.use_class_weights else None
            criterion = nn.CrossEntropyLoss(weight=weight)

            train_ds = VideoClipDataset(X_tr, y_tr, clip_len=self.clip_len, frame_size=self.frame_size,
                                         train=True, augment=self.augment)
            train_loader = _make_loader(train_ds, self.batch_size, shuffle=True)
            val_loader = None
            if len(X_val):
                val_ds = VideoClipDataset(X_val, y_val, clip_len=self.clip_len, frame_size=self.frame_size, train=False)
                val_loader = _make_loader(val_ds, self.batch_size, shuffle=False)

            best_val_loss = float("inf")
            best_state = None
            epochs_no_improve = 0
            for epoch in range(self.epochs):
                self._cnn.train(); self._lstm.train(); self._classifier.train()
                total_loss = 0
                for clips, labels in train_loader:
                    clips = _to_device(clips, self._device)
                    labels = _to_device(labels, self._device)
                    optimizer.zero_grad()
                    with self._amp.autocast():
                        loss = criterion(self._forward(clips), labels)
                    self._amp.backward(loss)
                    self._amp.step(optimizer)
                    total_loss += loss.item()
                msg = f"    Epoch {epoch+1}/{self.epochs}  train_loss={total_loss/len(train_loader):.4f}"

                if val_loader is not None:
                    val_loss = self._eval_loss(val_loader, criterion)
                    msg += f"  val_loss={val_loss:.4f}"
                    if val_loss < best_val_loss - 1e-4:
                        best_val_loss = val_loss
                        best_state = (copy.deepcopy(self._cnn.state_dict()),
                                      copy.deepcopy(self._lstm.state_dict()),
                                      copy.deepcopy(self._classifier.state_dict()))
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    print(msg)
                    if epochs_no_improve >= self.patience:
                        print(f"    Early stopping at epoch {epoch+1} (patience={self.patience})")
                        break
                else:
                    print(msg)

            if best_state is not None:
                self._cnn.load_state_dict(best_state[0])
                self._lstm.load_state_dict(best_state[1])
                self._classifier.load_state_dict(best_state[2])
            return self

        def _eval_loss(self, loader, criterion):
            self._cnn.eval(); self._lstm.eval(); self._classifier.eval()
            total, n = 0.0, 0
            with torch.no_grad():
                for clips, labels in loader:
                    clips = _to_device(clips, self._device)
                    labels = _to_device(labels, self._device)
                    with self._amp.autocast():
                        loss = criterion(self._forward(clips), labels)
                    total += loss.item() * len(labels)
                    n += len(labels)
            return total / max(n, 1)

        def predict(self, X):
            if self._classifier is None: raise ValueError("Model not fitted")
            self._cnn.eval(); self._lstm.eval(); self._classifier.eval()
            dataset = VideoClipDataset(X, [0]*len(X), clip_len=self.clip_len, frame_size=self.frame_size)
            loader = _make_loader(dataset, self.batch_size, shuffle=False)
            preds = []
            with torch.no_grad():
                for clips, _ in loader:
                    clips = _to_device(clips, self._device)
                    with self._amp.autocast():
                        out = self._forward(clips)
                    preds.extend(out.argmax(dim=1).cpu().numpy())
            return self._le.inverse_transform(np.array(preds))

        @property
        def classes_(self): return self._classes

    @register("model", "cnn3d")
    class CNN3DModel(Model):
        def __init__(self, backbone="r3d_18", pretrained=False, lr=0.0005, epochs=20, batch_size=4,
                     random_state=42, clip_len=16, frame_size=(112, 112),
                     val_size=0.15, patience=5, use_class_weights=True, augment=True):
            self.backbone = backbone
            self.pretrained = pretrained
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.random_state = random_state
            self.clip_len = clip_len
            self.frame_size = frame_size
            self.val_size = val_size
            self.patience = patience
            self.use_class_weights = use_class_weights
            self.augment = augment
            self._model = None
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._amp = _AmpHelper(self._device)
            self._le = LabelEncoder()
            self._classes = None

        def _build(self, n_classes):
            model = tv_models.video.r3d_18(pretrained=self.pretrained)
            model.fc = nn.Linear(512, n_classes)
            return model.to(self._device)

        def fit(self, X, y):
            self._le.fit(y)
            y_enc = self._le.transform(y)
            self._classes = self._le.classes_
            self._model = self._build(len(self._classes))
            X_tr, X_val, y_tr, y_val = _train_val_split_paths(X, y_enc, val_size=self.val_size, seed=self.random_state)
            optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
            weight = _compute_class_weights(y_tr, len(self._classes)).to(self._device) if self.use_class_weights else None
            criterion = nn.CrossEntropyLoss(weight=weight)

            train_ds = VideoClipDataset(X_tr, y_tr, clip_len=self.clip_len, frame_size=self.frame_size,
                                         train=True, augment=self.augment)
            train_loader = _make_loader(train_ds, self.batch_size, shuffle=True)
            val_loader = None
            if len(X_val):
                val_ds = VideoClipDataset(X_val, y_val, clip_len=self.clip_len, frame_size=self.frame_size, train=False)
                val_loader = _make_loader(val_ds, self.batch_size, shuffle=False)

            best_val_loss = float("inf")
            best_state = None
            epochs_no_improve = 0
            for epoch in range(self.epochs):
                self._model.train()
                total_loss = 0
                for clips, labels in train_loader:
                    clips = _to_device(clips.permute(0, 2, 1, 3, 4), self._device)
                    labels = _to_device(labels, self._device)
                    optimizer.zero_grad()
                    with self._amp.autocast():
                        loss = criterion(self._model(clips), labels)
                    self._amp.backward(loss)
                    self._amp.step(optimizer)
                    total_loss += loss.item()
                msg = f"    Epoch {epoch+1}/{self.epochs}  train_loss={total_loss/len(train_loader):.4f}"

                if val_loader is not None:
                    val_loss = self._eval_loss(val_loader, criterion)
                    msg += f"  val_loss={val_loss:.4f}"
                    if val_loss < best_val_loss - 1e-4:
                        best_val_loss = val_loss
                        best_state = copy.deepcopy(self._model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    print(msg)
                    if epochs_no_improve >= self.patience:
                        print(f"    Early stopping at epoch {epoch+1} (patience={self.patience})")
                        break
                else:
                    print(msg)

            if best_state is not None:
                self._model.load_state_dict(best_state)
            return self

        def _eval_loss(self, loader, criterion):
            self._model.eval()
            total, n = 0.0, 0
            with torch.no_grad():
                for clips, labels in loader:
                    clips = _to_device(clips.permute(0, 2, 1, 3, 4), self._device)
                    labels = _to_device(labels, self._device)
                    with self._amp.autocast():
                        loss = criterion(self._model(clips), labels)
                    total += loss.item() * len(labels)
                    n += len(labels)
            return total / max(n, 1)

        def predict(self, X):
            if self._model is None: raise ValueError("Model not fitted")
            self._model.eval()
            dataset = VideoClipDataset(X, [0]*len(X), clip_len=self.clip_len, frame_size=self.frame_size)
            loader = _make_loader(dataset, self.batch_size, shuffle=False)
            preds = []
            with torch.no_grad():
                for clips, _ in loader:
                    clips = _to_device(clips.permute(0, 2, 1, 3, 4), self._device)
                    with self._amp.autocast():
                        out = self._model(clips)
                    preds.extend(out.argmax(dim=1).cpu().numpy())
            return self._le.inverse_transform(np.array(preds))

        @property
        def classes_(self): return self._classes
