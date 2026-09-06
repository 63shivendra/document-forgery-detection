import os
import io
import lmdb
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .augmentations import DocumentAugmenter

class DocTamperLMDBDataset(Dataset):
    """
    Official DocTamper LMDB PyTorch Dataset Reader.
    Reads 120,000+ document image-mask pairs directly from LMDB databases (`data.mdb`).
    Key format: `image-{index:09d}` and `label-{index:09d}`
    """
    def __init__(self, lmdb_dir, img_size=(256, 256), is_train=True):
        self.lmdb_dir = lmdb_dir
        self.img_size = img_size
        self.is_train = is_train
        self.augmenter = DocumentAugmenter(img_size=img_size)

        self.env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            num_samples_bytes = txn.get(b'num-samples')
            if num_samples_bytes:
                self.num_samples = int(num_samples_bytes.decode('utf-8'))
            else:
                self.num_samples = (txn.stat()['entries'] - 1) // 2

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        key_img = f"image-{idx:09d}".encode('utf-8')
        key_label = f"label-{idx:09d}".encode('utf-8')

        with self.env.begin(write=False) as txn:
            img_bytes = txn.get(key_img)
            label_bytes = txn.get(key_label)

        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        mask = Image.open(io.BytesIO(label_bytes)).convert('L')

        img_tensor, mask_tensor = self.augmenter(image, mask)
        return img_tensor, mask_tensor, f"doc_{idx:09d}"


class COCOGLIDEDataset(Dataset):
    """
    COCOGLIDE / TruFor Image Splicing & AI Inpainting Dataset Loader.
    Loads from `fake/` images and `mask/` ground truth masks.
    """
    def __init__(self, data_root, img_size=(256, 256), is_train=True, val_split=0.2):
        self.data_root = data_root
        self.augmenter = DocumentAugmenter(img_size=img_size)
        self.fake_dir = os.path.join(data_root, 'fake')
        self.mask_dir = os.path.join(data_root, 'mask')
        self.is_train = is_train

        all_images = sorted([f for f in os.listdir(self.fake_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        all_masks = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        split_idx = int(len(all_images) * (1.0 - val_split))
        if is_train:
            self.images = all_images[:split_idx]
            self.masks = all_masks[:split_idx]
        else:
            self.images = all_images[split_idx:]
            self.masks = all_masks[split_idx:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.fake_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if idx < len(self.masks):
            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            mask = Image.open(mask_path).convert('L')
        else:
            mask = Image.new('L', image.size, 0)

        img_tensor, mask_tensor = self.augmenter(image, mask)
        return img_tensor, mask_tensor, img_name


class SplicingDataset(Dataset):
    """
    Image Forgery Detection Splicing Dataset Loader.
    Loads from `tampered/` (or `real/`) and `ground truth/`.
    """
    def __init__(self, data_root, img_size=(256, 256), is_train=True, val_split=0.2):
        self.data_root = data_root
        self.augmenter = DocumentAugmenter(img_size=img_size)
        self.tampered_dir = os.path.join(data_root, 'tampered')
        self.gt_dir = os.path.join(data_root, 'ground truth')
        self.is_train = is_train

        if os.path.exists(self.tampered_dir):
            all_images = sorted([f for f in os.listdir(self.tampered_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        else:
            all_images = []

        # Deterministic 80/20 Train/Val Split
        split_idx = int(len(all_images) * (1.0 - val_split))
        if is_train:
            self.images = all_images[:split_idx]
            self.indices = list(range(0, split_idx))
        else:
            self.images = all_images[split_idx:]
            self.indices = list(range(split_idx, len(all_images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        global_idx = self.indices[idx]
        img_path = os.path.join(self.tampered_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Robust GT mask matching across multiple extensions
        mask = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
            gt_filename = f"gt_{global_idx+1:03d}{ext}"
            gt_path = os.path.join(self.gt_dir, gt_filename)
            if os.path.exists(gt_path):
                mask = Image.open(gt_path).convert('L')
                break

        if mask is None:
            mask = Image.new('L', image.size, 0)

        img_tensor, mask_tensor = self.augmenter(image, mask)
        return img_tensor, mask_tensor, img_name


class CASIADataset(Dataset):
    """
    CASIA v2.0 Image Tampering Detection Dataset Loader.
    Supports tampered images (splicing + copy-move) and ground truth masks.
    Automatically resolves subdirectories (CASIA 2.0 Image Tampering Detection Dataset, Tp, Gt, CASIA2.0_Tampered, etc.).
    """
    def __init__(self, data_root, img_size=(256, 256), is_train=True, val_split=0.2):
        self.data_root = data_root
        self.augmenter = DocumentAugmenter(img_size=img_size)
        self.is_train = is_train

        # Locate tampered images directory
        possible_tp_dirs = [
            os.path.join(data_root, 'CASIA2', 'Tp'),
            os.path.join(data_root, 'Tp'),
            os.path.join(data_root, 'CASIA2.0_Tampered'),
            os.path.join(data_root, 'tampered'),
            os.path.join(data_root, 'Tampered'),
            data_root
        ]

        self.tp_dir = None
        for d in possible_tp_dirs:
            if os.path.exists(d) and any(f.lower().endswith(('.jpg', '.png', '.tif', '.bmp', '.jpeg')) for f in os.listdir(d)):
                self.tp_dir = d
                break

        if self.tp_dir:
            all_images = sorted([f for f in os.listdir(self.tp_dir) if f.lower().endswith(('.jpg', '.png', '.tif', '.bmp', '.jpeg'))])
        else:
            all_images = []

        # Locate ground truth directory
        possible_gt_dirs = [
            os.path.join(data_root, 'CASIA2', 'CASIA 2 Groundtruth'),
            os.path.join(data_root, 'CASIA2.0_Groundtruth'),
            os.path.join(data_root, 'Gt'),
            os.path.join(data_root, 'gt'),
            os.path.join(data_root, 'groundtruth'),
            os.path.join(data_root, 'Groundtruth'),
            self.tp_dir
        ]
        self.gt_dir = None
        for d in possible_gt_dirs:
            if d and os.path.exists(d):
                self.gt_dir = d
                break

        split_idx = int(len(all_images) * (1.0 - val_split))
        if is_train:
            self.images = all_images[:split_idx]
        else:
            self.images = all_images[split_idx:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.tp_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Find matching GT mask
        base_name = os.path.splitext(img_name)[0]
        mask = None

        if self.gt_dir:
            gt_candidates = [
                f"{base_name}_gt.png",
                f"{base_name}_gt.jpg",
                f"{base_name}.png",
                f"{base_name}.jpg",
                f"{base_name}_gt.tif"
            ]
            for candidate in gt_candidates:
                gt_path = os.path.join(self.gt_dir, candidate)
                if os.path.exists(gt_path):
                    mask = Image.open(gt_path).convert('L')
                    break

            if mask is None:
                # Suffix ID fallback matching (e.g. _00306_gt.png)
                parts = base_name.split('_')
                if len(parts) >= 2:
                    suffix_id = parts[-1]
                    suffix_matches = [g for g in os.listdir(self.gt_dir) if g.endswith(f"_{suffix_id}_gt.png") or g.endswith(f"_{suffix_id}_gt.jpg")]
                    if len(suffix_matches) > 0:
                        gt_path = os.path.join(self.gt_dir, suffix_matches[0])
                        mask = Image.open(gt_path).convert('L')

        if mask is None:
            mask = Image.new('L', image.size, 0)

        img_tensor, mask_tensor = self.augmenter(image, mask)
        return img_tensor, mask_tensor, img_name


from torch.utils.data import Dataset, ConcatDataset

def get_dataset(dataset_name, data_root="./data", split="train", img_size=(256, 256)):
    """
    Universal Factory Function to return the appropriate Dataset instance.
    Supports individual datasets ('doctamper', 'cocoglide', 'casia', 'splicing')
    and unified combined training ('combined' / 'splicing_combined').
    """
    dataset_name = dataset_name.lower()

    if 'combined' in dataset_name or 'all_splicing' in dataset_name or 'splicing_combined' in dataset_name:
        casia_path      = os.path.join(data_root, 'CASIA 2.0 Image Tampering Detection Dataset')
        cocoglide_path  = os.path.join(data_root, 'COCOGLIDE_trufor')
        splicing_path   = os.path.join(data_root, 'Image Forgery detecion dataset (splicing)')
        # DocTamper LMDB: 120k document forgery images — biggest single quality boost
        doctamper_train = os.path.join(data_root, 'Doctamper', 'DocTamperV1-TrainingSet')
        doctamper_test  = os.path.join(data_root, 'Doctamper', 'DocTamperV1-TestingSet')

        datasets = []
        if os.path.exists(casia_path):
            datasets.append(CASIADataset(data_root=casia_path, img_size=img_size, is_train=(split=='train')))
        if os.path.exists(cocoglide_path):
            datasets.append(COCOGLIDEDataset(data_root=cocoglide_path, img_size=img_size, is_train=(split=='train')))
        if os.path.exists(splicing_path):
            datasets.append(SplicingDataset(data_root=splicing_path, img_size=img_size, is_train=(split=='train')))

        # Add DocTamper LMDB: use TrainingSet for train split, TestingSet for val split
        if split == 'train' and os.path.exists(doctamper_train):
            datasets.append(DocTamperLMDBDataset(lmdb_dir=doctamper_train, img_size=img_size, is_train=True))
        elif split != 'train' and os.path.exists(doctamper_test):
            datasets.append(DocTamperLMDBDataset(lmdb_dir=doctamper_test, img_size=img_size, is_train=False))

        return ConcatDataset(datasets)


    elif 'doctamper' in dataset_name:
        if split == 'train':
            lmdb_path = os.path.join(data_root, 'Doctamper', 'DocTamperV1-TrainingSet')
        else:
            lmdb_path = os.path.join(data_root, 'Doctamper', 'DocTamperV1-TestingSet')
        return DocTamperLMDBDataset(lmdb_dir=lmdb_path, img_size=img_size, is_train=(split=='train'))

    elif 'cocoglide' in dataset_name or 'trufor' in dataset_name:
        cocoglide_path = os.path.join(data_root, 'COCOGLIDE_trufor')
        return COCOGLIDEDataset(data_root=cocoglide_path, img_size=img_size, is_train=(split=='train'))

    elif 'casia' in dataset_name:
        casia_path = os.path.join(data_root, 'CASIA 2.0 Image Tampering Detection Dataset')
        if not os.path.exists(casia_path):
            casia_path = os.path.join(data_root, 'CASIA2.0')
        return CASIADataset(data_root=casia_path, img_size=img_size, is_train=(split=='train'))

    elif 'splicing' in dataset_name:
        splicing_path = os.path.join(data_root, 'Image Forgery detecion dataset (splicing)')
        return SplicingDataset(data_root=splicing_path, img_size=img_size, is_train=(split=='train'))

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
