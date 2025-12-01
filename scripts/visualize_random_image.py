"""
Ki-67 Single Image Visualization
===============================
Select a random test image and visualize model predictions vs ground truth
"""

import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as scipy_label
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

# ============================================================================
# MODEL AND DATASET CLASSES (from training script)
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

class Ki67PointDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None, heatmap_generator=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.heatmap_generator = heatmap_generator or ImprovedPointHeatmapGenerator(sigma=8.0)

        self.image_files = sorted([f for f in os.listdir(image_dir)
                                   if f.endswith('.png')])

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

def get_validation_augmentation():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

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

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def load_ground_truth_points(annotation_dir, image_name):
    """Load ground truth points for an image"""
    base_name = os.path.splitext(image_name)[0]

    pos_h5 = os.path.join(annotation_dir, 'positive', f"{base_name}.h5")
    neg_h5 = os.path.join(annotation_dir, 'negative', f"{base_name}.h5")

    pos_points = []
    neg_points = []

    # Load positive points
    try:
        with h5py.File(pos_h5, 'r') as f:
            if 'coordinates' in f:
                pos_points = f['coordinates'][:]
    except:
        pos_points = []

    # Load negative points
    try:
        with h5py.File(neg_h5, 'r') as f:
            if 'coordinates' in f:
                neg_points = f['coordinates'][:]
    except:
        neg_points = []

    return np.array(pos_points), np.array(neg_points)

def predict_points_from_heatmaps(heatmaps, threshold=0.3):
    """Extract points from predicted heatmaps"""
    pos_heatmap = heatmaps[0]
    neg_heatmap = heatmaps[1]

    # Detect peaks
    pos_peaks = peak_local_max(pos_heatmap, threshold_abs=threshold, min_distance=10)
    neg_peaks = peak_local_max(neg_heatmap, threshold_abs=threshold, min_distance=10)

    return pos_peaks, neg_peaks

def visualize_single_image_predictions(image_path, model, transform, device='cpu'):
    """Visualize predictions for a single image"""

    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()

    # Apply transforms
    if transform:
        transformed = transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(device)
    else:
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(device)

    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        pred_heatmaps = torch.sigmoid(outputs).squeeze(0).cpu().numpy()

    # Extract predicted points
    pred_pos_points, pred_neg_points = predict_points_from_heatmaps(pred_heatmaps)

    # Load ground truth points
    image_name = os.path.basename(image_path)
    gt_pos_points, gt_neg_points = load_ground_truth_points(
        'BCData/annotations/test', image_name
    )

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Ki-67 Cell Detection: {image_name}', fontsize=16, fontweight='bold')

    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')

    # Ground truth positive cells
    axes[0, 1].imshow(original_image)
    if len(gt_pos_points) > 0:
        axes[0, 1].scatter(gt_pos_points[:, 0], gt_pos_points[:, 1],
                          c='red', s=50, alpha=0.8, edgecolors='white', linewidth=2)
    axes[0, 1].set_title(f'Ground Truth Positive Cells ({len(gt_pos_points)})', fontsize=14)
    axes[0, 1].axis('off')

    # Ground truth negative cells
    axes[0, 2].imshow(original_image)
    if len(gt_neg_points) > 0:
        axes[0, 2].scatter(gt_neg_points[:, 0], gt_neg_points[:, 1],
                          c='blue', s=50, alpha=0.8, edgecolors='white', linewidth=2)
    axes[0, 2].set_title(f'Ground Truth Negative Cells ({len(gt_neg_points)})', fontsize=14)
    axes[0, 2].axis('off')

    # Predicted positive heatmap
    axes[1, 0].imshow(pred_heatmaps[0], cmap='hot')
    axes[1, 0].set_title('Predicted Positive Heatmap', fontsize=14)
    axes[1, 0].axis('off')

    # Predicted negative heatmap
    axes[1, 1].imshow(pred_heatmaps[1], cmap='cool')
    axes[1, 1].set_title('Predicted Negative Heatmap', fontsize=14)
    axes[1, 1].axis('off')

    # Combined prediction overlay
    axes[1, 2].imshow(original_image, alpha=0.7)

    # Plot predicted points
    if len(pred_pos_points) > 0:
        axes[1, 2].scatter(pred_pos_points[:, 1], pred_pos_points[:, 0],
                          c='red', s=60, marker='x', alpha=0.9, linewidth=3, label='Pred Positive')
    if len(pred_neg_points) > 0:
        axes[1, 2].scatter(pred_neg_points[:, 1], pred_neg_points[:, 0],
                          c='blue', s=60, marker='x', alpha=0.9, linewidth=3, label='Pred Negative')

    # Plot ground truth points for comparison
    if len(gt_pos_points) > 0:
        axes[1, 2].scatter(gt_pos_points[:, 0], gt_pos_points[:, 1],
                          c='orange', s=40, marker='o', alpha=0.7, edgecolors='red', linewidth=2, label='GT Positive')
    if len(gt_neg_points) > 0:
        axes[1, 2].scatter(gt_neg_points[:, 0], gt_neg_points[:, 1],
                          c='cyan', s=40, marker='o', alpha=0.7, edgecolors='blue', linewidth=2, label='GT Negative')

    axes[1, 2].set_title('Predictions vs Ground Truth Overlay', fontsize=14)
    axes[1, 2].legend(loc='upper right', fontsize=10)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f'ki67_visualization_{image_name.replace(".png", "")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Visualization saved as: ki67_visualization_{image_name.replace('.png', '')}.png")

    # Print statistics
    print(f"\n{'='*60}")
    print(f"📊 ANALYSIS FOR: {image_name}")
    print(f"{'='*60}")

    print("GROUND TRUTH:")
    print(f"  Positive cells: {len(gt_pos_points)}")
    print(f"  Negative cells: {len(gt_neg_points)}")
    print(f"  Total cells: {len(gt_pos_points) + len(gt_neg_points)}")

    print("\nPREDICTIONS:")
    print(f"  Positive cells: {len(pred_pos_points)}")
    print(f"  Negative cells: {len(pred_neg_points)}")
    print(f"  Total cells: {len(pred_pos_points) + len(pred_neg_points)}")

    # Classification
    gt_malignant = len(gt_pos_points) > len(gt_neg_points)
    pred_malignant = len(pred_pos_points) > len(pred_neg_points)

    print("\nCLASSIFICATION:")
    print(f"  Ground Truth: {'MALIGNANT' if gt_malignant else 'BENIGN'}")
    print(f"  Prediction: {'MALIGNANT' if pred_malignant else 'BENIGN'}")
    print(f"  Correct: {'✅' if gt_malignant == pred_malignant else '❌'}")

    return {
        'image_name': image_name,
        'gt_positive': len(gt_pos_points),
        'gt_negative': len(gt_neg_points),
        'pred_positive': len(pred_pos_points),
        'pred_negative': len(pred_neg_points),
        'gt_malignant': gt_malignant,
        'pred_malignant': pred_malignant,
        'correct': gt_malignant == pred_malignant
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("🔬 Ki-67 Single Image Visualization")
    print("="*50)

    # Configuration
    CHECKPOINT_PATH = 'ki67-point-epoch=68-val_peak_f1_avg=0.8503.ckpt'
    TEST_IMAGE_DIR = 'BCData/images/test'

    # Check if files exist
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        exit(1)

    if not os.path.exists(TEST_IMAGE_DIR):
        print(f"❌ Test directory not found: {TEST_IMAGE_DIR}")
        exit(1)

    # Get list of test images
    test_images = [f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith('.png')]
    print(f"Found {len(test_images)} test images")

    # Select random image
    random_image = random.choice(test_images)
    image_path = os.path.join(TEST_IMAGE_DIR, random_image)

    print(f"🎲 Selected random image: {random_image}")
    print(f"📁 Full path: {image_path}")

    # Load model
    print("\nLoading model...")
    try:
        model = ImprovedKi67PointDetectionModel.load_from_checkpoint(CHECKPOINT_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        exit(1)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"✅ Using device: {device}")

    # Create transforms
    transform = get_validation_augmentation()

    # Run visualization
    print("\n🎨 Generating visualization...")
    try:
        results = visualize_single_image_predictions(image_path, model, transform, device)
        print("✅ Visualization completed!")
        print(f"\n📈 Results saved for image: {results['image_name']}")

    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*50}")
    print("🎉 Script Complete!")
    print(f"{'='*50}")