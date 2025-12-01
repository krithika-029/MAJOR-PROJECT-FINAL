"""
Ki-67 Model Validation Script - Colab Ready
===========================================
Validates the trained model and reports actual accuracy metrics

USAGE:
1. Upload this script to Colab
2. Upload your checkpoint: ki67-point-epoch=68-val_peak_f1_avg=0.8503.ckpt
3. Run the script
4. Get actual F1 scores and accuracy metrics
"""

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================

!pip install -q segmentation-models-pytorch albumentations h5py opencv-python-headless
!pip install -q pytorch-lightning torchmetrics scipy scikit-image

# ============================================================================
# GOOGLE DRIVE SETUP (for BCData.zip)
# ============================================================================

from google.colab import drive
drive.mount('/content/drive')

# Unzip BCData.zip from Google Drive
!unzip -q "/content/drive/MyDrive/BCData.zip" -d "/content/"
print("✓ Dataset extracted to /content/BCData")

# ============================================================================
# IMPORTS
# ============================================================================

import os
import h5py
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as scipy_label
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# ============================================================================
# IMPROVED POINT HEATMAP GENERATOR (from training script)
# ============================================================================

class ImprovedPointHeatmapGenerator:
    def __init__(self, sigma=8.0):
        self.sigma = sigma

    def generate_heatmap(self, points, image_shape=(640, 640)):
        if len(points) == 0:
            return np.zeros(image_shape, dtype=np.float32)

        heatmap = np.zeros(image_shape, dtype=np.float32)

        kernel_size = int(6 * self.sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = np.arange(0, kernel_size)
        y = x[:, np.newaxis]
        x0 = y0 = kernel_size // 2

        gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
        gaussian = gaussian / gaussian.max()

        for point in points:
            x, y = int(point[0]), int(point[1])

            x_min = max(0, x - kernel_size // 2)
            x_max = min(image_shape[1], x + kernel_size // 2 + 1)
            y_min = max(0, y - kernel_size // 2)
            y_max = min(image_shape[0], y + kernel_size // 2 + 1)

            k_x_min = max(0, kernel_size // 2 - x)
            k_x_max = min(kernel_size, kernel_size // 2 + (image_shape[1] - x))
            k_y_min = max(0, kernel_size // 2 - y)
            k_y_max = min(kernel_size, kernel_size // 2 + (image_shape[0] - y))

            heatmap[y_min:y_max, x_min:x_max] = np.maximum(
                heatmap[y_min:y_max, x_min:x_max],
                gaussian[k_y_min:k_y_max, k_x_min:k_x_max]
            )

        return heatmap

# ============================================================================
# DATASET (from training script)
# ============================================================================

class Ki67PointDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None, heatmap_generator=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.heatmap_generator = heatmap_generator or ImprovedPointHeatmapGenerator(sigma=8.0)

        self.image_files = sorted([f for f in os.listdir(image_dir)
                                   if f.endswith('.png')])

        print(f"Found {len(self.image_files)} images in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def load_points_from_h5(self, h5_path):
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'coordinates' in f:
                    return f['coordinates'][:]
        except:
            pass
        return np.array([])

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        base_name = os.path.splitext(img_name)[0]

        pos_h5 = os.path.join(self.annotation_dir, 'positive', f"{base_name}.h5")
        neg_h5 = os.path.join(self.annotation_dir, 'negative', f"{base_name}.h5")

        pos_points = self.load_points_from_h5(pos_h5)
        neg_points = self.load_points_from_h5(neg_h5)

        pos_heatmap = self.heatmap_generator.generate_heatmap(pos_points, image.shape[:2])
        neg_heatmap = self.heatmap_generator.generate_heatmap(neg_points, image.shape[:2])

        heatmaps = np.stack([pos_heatmap, neg_heatmap], axis=0)

        if self.transform:
            transformed = self.transform(
                image=image,
                mask=heatmaps.transpose(1, 2, 0)
            )
            image = transformed['image']
            heatmaps = transformed['mask'].permute(2, 0, 1).float()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            heatmaps = torch.from_numpy(heatmaps).float()

        return image, heatmaps, img_name

# ============================================================================
# AUGMENTATION (from training script)
# ============================================================================

def get_validation_augmentation():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# ============================================================================
# IMPROVED LOSS (from training script)
# ============================================================================

class ImprovedHeatmapLoss(nn.Module):
    def __init__(self, pos_weight=10.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
            reduction='mean'
        )

    def dice_loss(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        total_loss = 0.4 * bce_loss + 0.4 * dice + 0.2 * focal
        return total_loss, {
            'bce': bce_loss.item(),
            'dice': dice.item(),
            'focal': focal.item()
        }

# ============================================================================
# MODEL (from training script)
# ============================================================================

class ImprovedKi67PointDetectionModel(pl.LightningModule):
    def __init__(self, encoder_name='efficientnet-b3', learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=2,
            activation=None,
        )

        self.criterion = ImprovedHeatmapLoss(pos_weight=10.0)
        self.validation_outputs = []

    def forward(self, x):
        return self.model(x)

    def _compute_peak_metrics(self, pred_heatmaps, target_heatmaps, threshold=0.3, min_distance=10):
        """Compute F1 based on detected peaks"""
        total_tp_pos, total_fp_pos, total_fn_pos = 0, 0, 0
        total_tp_neg, total_fp_neg, total_fn_neg = 0, 0, 0

        for i in range(len(pred_heatmaps)):
            pred_pos = pred_heatmaps[i, 0].float().numpy()
            pred_neg = pred_heatmaps[i, 1].float().numpy()
            target_pos = target_heatmaps[i, 0].float().numpy()
            target_neg = target_heatmaps[i, 1].float().numpy()

            # Positive channel
            pred_peaks_pos = peak_local_max(pred_pos,
                                           threshold_abs=threshold,
                                           min_distance=min_distance)
            target_peaks_pos = peak_local_max(target_pos,
                                             threshold_abs=0.2,
                                             min_distance=min_distance)

            if len(pred_peaks_pos) > 0 and len(target_peaks_pos) > 0:
                distances = cdist(pred_peaks_pos, target_peaks_pos)
                matches = (distances < min_distance).any(axis=1)
                total_tp_pos += matches.sum()
                total_fp_pos += len(pred_peaks_pos) - matches.sum()
                total_fn_pos += len(target_peaks_pos) - matches.sum()
            else:
                total_fp_pos += len(pred_peaks_pos)
                total_fn_pos += len(target_peaks_pos)

            # Negative channel
            pred_peaks_neg = peak_local_max(pred_neg,
                                           threshold_abs=threshold,
                                           min_distance=min_distance)
            target_peaks_neg = peak_local_max(target_neg,
                                             threshold_abs=0.2,
                                             min_distance=min_distance)

            if len(pred_peaks_neg) > 0 and len(target_peaks_neg) > 0:
                distances = cdist(pred_peaks_neg, target_peaks_neg)
                matches = (distances < min_distance).any(axis=1)
                total_tp_neg += matches.sum()
                total_fp_neg += len(pred_peaks_neg) - matches.sum()
                total_fn_neg += len(target_peaks_neg) - matches.sum()
            else:
                total_fp_neg += len(pred_peaks_neg)
                total_fn_neg += len(target_peaks_neg)

        f1_pos = 2 * total_tp_pos / (2 * total_tp_pos + total_fp_pos + total_fn_pos + 1e-6)
        f1_neg = 2 * total_tp_neg / (2 * total_tp_neg + total_fp_neg + total_fn_neg + 1e-6)

        return {
            'pos': f1_pos,
            'neg': f1_neg,
            'avg': (f1_pos + f1_neg) / 2
        }

# ============================================================================
# VALIDATION FUNCTION
# ============================================================================

def validate_model_comprehensive(checkpoint_path, data_path='/content/BCData', batch_size=8):
    """
    Comprehensive validation of the trained model
    Returns detailed accuracy metrics
    """

    print(f"\n{'='*80}")
    print("🔍 COMPREHENSIVE MODEL VALIDATION")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {data_path}")
    print(f"Batch Size: {batch_size}")
    print(f"{'='*80}\n")

    # Check if files exist
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None

    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        return None

    # Load model
    print("Loading model...")
    try:
        model = ImprovedKi67PointDetectionModel.load_from_checkpoint(checkpoint_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"✓ Using device: {device}")

    # Create dataset
    heatmap_gen = ImprovedPointHeatmapGenerator(sigma=8.0)

    try:
        val_dataset = Ki67PointDataset(
            image_dir=os.path.join(data_path, 'images/validation'),
            annotation_dir=os.path.join(data_path, 'annotations/validation'),
            transform=get_validation_augmentation(),
            heatmap_generator=heatmap_gen
        )
        print(f"✓ Validation dataset created: {len(val_dataset)} images")
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return None

    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Validation loop
    print("\nRunning validation...")
    all_pred_heatmaps = []
    all_target_heatmaps = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            images, heatmaps, names = batch
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            # Forward pass
            outputs = model(images)
            loss, _ = model.criterion(outputs, heatmaps)

            # Store results
            pred_heatmaps = torch.sigmoid(outputs)
            all_pred_heatmaps.append(pred_heatmaps.cpu())
            all_target_heatmaps.append(heatmaps.cpu())

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 10 == 0:
                print(f"Processed {num_batches * batch_size} images...")

    # Aggregate results
    all_pred = torch.cat(all_pred_heatmaps)
    all_target = torch.cat(all_target_heatmaps)
    avg_loss = total_loss / num_batches

    print(f"✓ Validation complete: {len(all_pred)} images processed")

    # Compute comprehensive metrics
    print("\nComputing metrics...")

    # 1. Pixel-wise metrics
    pred_threshold = 0.3
    target_threshold = 0.2

    pred_binary = (all_pred > pred_threshold).float()
    target_binary = (all_target > target_threshold).float()

    pos_pred = pred_binary[:, 0].flatten().numpy()
    pos_target = target_binary[:, 0].flatten().numpy()
    neg_pred = pred_binary[:, 1].flatten().numpy()
    neg_target = target_binary[:, 1].flatten().numpy()

    pixel_f1_pos = f1_score(pos_target, pos_pred, zero_division=0)
    pixel_f1_neg = f1_score(neg_target, neg_pred, zero_division=0)
    pixel_f1_avg = (pixel_f1_pos + pixel_f1_neg) / 2

    pixel_precision_pos = precision_score(pos_target, pos_pred, zero_division=0)
    pixel_precision_neg = precision_score(neg_target, neg_pred, zero_division=0)
    pixel_recall_pos = recall_score(pos_target, pos_pred, zero_division=0)
    pixel_recall_neg = recall_score(neg_target, neg_pred, zero_division=0)

    # 2. Peak-based metrics (more relevant)
    peak_metrics = model._compute_peak_metrics(all_pred, all_target, threshold=0.3, min_distance=10)

    # 3. Count-based metrics
    total_predicted_pos = 0
    total_predicted_neg = 0
    total_ground_truth_pos = 0
    total_ground_truth_neg = 0

    for i in range(len(all_pred)):
        pred_pos = all_pred[i, 0].float().numpy()
        pred_neg = all_pred[i, 1].float().numpy()
        target_pos = all_target[i, 0].float().numpy()
        target_neg = all_target[i, 1].float().numpy()

        pred_peaks_pos = peak_local_max(pred_pos, threshold_abs=0.3, min_distance=10)
        pred_peaks_neg = peak_local_max(pred_neg, threshold_abs=0.3, min_distance=10)
        target_peaks_pos = peak_local_max(target_pos, threshold_abs=0.2, min_distance=10)
        target_peaks_neg = peak_local_max(target_neg, threshold_abs=0.2, min_distance=10)

        total_predicted_pos += len(pred_peaks_pos)
        total_predicted_neg += len(pred_peaks_neg)
        total_ground_truth_pos += len(target_peaks_pos)
        total_ground_truth_neg += len(target_peaks_neg)

    # Results
    results = {
        'validation_loss': avg_loss,
        'pixel_f1_avg': pixel_f1_avg,
        'pixel_f1_positive': pixel_f1_pos,
        'pixel_f1_negative': pixel_f1_neg,
        'pixel_precision_pos': pixel_precision_pos,
        'pixel_precision_neg': pixel_precision_neg,
        'pixel_recall_pos': pixel_recall_pos,
        'pixel_recall_neg': pixel_recall_neg,
        'peak_f1_avg': peak_metrics['avg'],
        'peak_f1_pos': peak_metrics['pos'],
        'peak_f1_neg': peak_metrics['neg'],
        'total_predicted_positive': total_predicted_pos,
        'total_predicted_negative': total_predicted_neg,
        'total_ground_truth_positive': total_ground_truth_pos,
        'total_ground_truth_negative': total_ground_truth_neg,
        'num_images': len(all_pred)
    }

    # Print results
    print(f"\n{'='*80}")
    print("📊 VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"Validation Loss: {results['validation_loss']:.4f}")
    print()
    print("PIXEL-WISE METRICS:")
    print(f"  F1 Average:     {results['pixel_f1_avg']:.4f}")
    print(f"  F1 Positive:    {results['pixel_f1_positive']:.4f} (P: {results['pixel_precision_pos']:.4f}, R: {results['pixel_recall_pos']:.4f})")
    print(f"  F1 Negative:    {results['pixel_f1_negative']:.4f} (P: {results['pixel_precision_neg']:.4f}, R: {results['pixel_recall_neg']:.4f})")
    print()
    print("PEAK-BASED METRICS (More Relevant for Point Detection):")
    print(f"  Peak F1 Average: {results['peak_f1_avg']:.4f}")
    print(f"  Peak F1 Positive: {results['peak_f1_pos']:.4f}")
    print(f"  Peak F1 Negative: {results['peak_f1_neg']:.4f}")
    print()
    print("CELL COUNTS:")
    print(f"  Predicted Positive: {results['total_predicted_positive']} cells")
    print(f"  Ground Truth Positive: {results['total_ground_truth_positive']} cells")
    print(f"  Predicted Negative: {results['total_predicted_negative']} cells")
    print(f"  Ground Truth Negative: {results['total_ground_truth_negative']} cells")
    print(f"  Total Images: {results['num_images']}")
    print()
    print("ACCURACY SUMMARY:")
    accuracy = results['peak_f1_avg'] * 100
    print(f"  Overall Accuracy: {accuracy:.2f}%")
    if accuracy >= 90:
        print("  🎉 EXCELLENT: 90%+ accuracy achieved!")
    elif accuracy >= 80:
        print("  ✅ GOOD: 80%+ accuracy - clinical grade")
    elif accuracy >= 70:
        print("  ⚠️  FAIR: 70%+ accuracy - needs improvement")
    else:
        print("  ❌ POOR: <70% accuracy - retrain needed")
    print(f"{'='*80}\n")

    return results

# ============================================================================
# VISUALIZATION FUNCTION
# ============================================================================

def visualize_predictions(checkpoint_path, image_path, save_path=None):
    """Visualize predictions on a single image"""

    # Load model
    model = ImprovedKi67PointDetectionModel.load_from_checkpoint(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = image_rgb.copy()

    # Preprocess
    transform = get_validation_augmentation()
    transformed = transform(image=image_rgb)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        heatmaps = torch.sigmoid(output).squeeze().cpu().numpy()

    # Detect points
    pos_points = peak_local_max(heatmaps[0], min_distance=8, threshold_abs=0.3)
    neg_points = peak_local_max(heatmaps[1], min_distance=8, threshold_abs=0.3)

    # Visualization
    detection_img = original.copy()
    for point in pos_points:
        cv2.circle(detection_img, (point[1], point[0]), 5, (0, 255, 0), -1)  # Green for positive
        cv2.circle(detection_img, (point[1], point[0]), 7, (255, 255, 255), 1)
    for point in neg_points:
        cv2.circle(detection_img, (point[1], point[0]), 5, (255, 0, 0), -1)  # Red for negative
        cv2.circle(detection_img, (point[1], point[0]), 7, (255, 255, 255), 1)

    # Calculate Ki-67 index
    num_positive = len(pos_points)
    num_negative = len(neg_points)
    ki67_index = num_positive / (num_positive + num_negative + 1e-10) * 100

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(heatmaps[0], cmap='hot')
    axes[0, 1].set_title(f'Positive Heatmap\nMax: {heatmaps[0].max():.3f}', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(heatmaps[1], cmap='hot')
    axes[0, 2].set_title(f'Negative Heatmap\nMax: {heatmaps[1].max():.3f}', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(detection_img)
    axes[1, 0].set_title(f'Detected Points\nPos: {num_positive} | Neg: {num_negative}',
                        fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(heatmaps[0] > 0.3, cmap='gray')
    axes[1, 1].set_title('Positive Detection', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(heatmaps[1] > 0.3, cmap='gray')
    axes[1, 2].set_title('Negative Detection', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')

    fig.suptitle(f'Ki-67 Detection Results - F1 Score: {0.8503:.1%}\nKi-67 Index: {ki67_index:.2f}%',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    plt.show()

    return pos_points, neg_points, {'positive': num_positive, 'negative': num_negative, 'ki67_index': ki67_index}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("🎯 Ki-67 Model Validation Script")
    print("="*80)

    # Configuration
    CHECKPOINT_PATH = '/content/ki67-point-epoch=68-val_peak_f1_avg=0.8503.ckpt'
    DATA_PATH = '/content/BCData'  # Dataset extracted from Google Drive
    BATCH_SIZE = 8

    # Run validation
    results = validate_model_comprehensive(
        checkpoint_path=CHECKPOINT_PATH,
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE
    )

    if results:
        print("✅ Validation completed successfully!")
        print(f"📊 Key Results:")
        print(f"   Peak F1 Score: {results['peak_f1_avg']:.4f} ({results['peak_f1_avg']*100:.1f}%)")
        print(f"   Total Images: {results['num_images']}")
        print(f"   Positive Cells Detected: {results['total_predicted_positive']}")
        print(f"   Ground Truth Positive: {results['total_ground_truth_positive']}")

        # Visualize a sample
        print("\n🎨 Generating visualization...")
        try:
            test_image = '/content/BCData/images/test/0.png'
            if os.path.exists(test_image):
                pos_pts, neg_pts, stats = visualize_predictions(CHECKPOINT_PATH, test_image)
                print(f"✓ Sample visualization complete")
                print(f"  Detected {stats['positive']} positive, {stats['negative']} negative cells")
                print(f"  Ki-67 Index: {stats['ki67_index']:.2f}%")
            else:
                print("⚠️ Test image not found, skipping visualization")
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")

    else:
        print("❌ Validation failed - check paths and files")

    print(f"\n{'='*80}")
    print("🎉 Script Complete!")
    print(f"{'='*80}")