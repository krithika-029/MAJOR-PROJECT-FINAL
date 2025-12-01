"""
Ki-67 MALIGNANCY CLASSIFICATION VALIDATION
===========================================
Classify test images as MALIGNANT or BENIGN based on Ki-67 cell counts
Compare with ground truth and generate confusion matrix

USAGE:
1. Upload this script to Colab
2. Upload your checkpoint: ki67-point-epoch=68-val_peak_f1_avg=0.8503.ckpt
3. Run the script
4. Get classification accuracy and confusion matrix
"""

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================

!pip install -q segmentation-models-pytorch albumentations h5py opencv-python-headless
!pip install -q pytorch-lightning torchmetrics scipy scikit-image seaborn tqdm

# ============================================================================
# GOOGLE DRIVE SETUP (for BCData.zip)
# ============================================================================

from google.colab import drive
drive.mount('/content/drive')

# Check if dataset already exists, otherwise extract
import os
if not os.path.exists('/content/BCData'):
    print("Extracting BCData.zip from Google Drive...")
    !unzip -q "/content/drive/MyDrive/BCData.zip" -d "/content/"
    print("✓ Dataset extracted to /content/BCData")
else:
    print("✓ Dataset already exists at /content/BCData")

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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as scipy_label
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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

        # Add the criterion to match the saved checkpoint
        self.criterion = ImprovedHeatmapLoss(pos_weight=10.0)
        self.validation_outputs = []

    def forward(self, x):
        return self.model(x)

# ============================================================================
# MALIGNANCY CLASSIFICATION FUNCTION
# ============================================================================

def classify_malignancy_from_cells(pos_count, neg_count, threshold_ratio=1.0):
    """
    Classify image as malignant or benign based on cell counts

    Args:
        pos_count: Number of positive (Ki-67+) cells
        neg_count: Number of negative (Ki-67-) cells
        threshold_ratio: If pos/neg > threshold_ratio, classify as malignant

    Returns:
        'malignant' or 'benign', ki67_index
    """
    total_cells = pos_count + neg_count
    if total_cells == 0:
        return 'benign', 0.0  # No cells = benign

    ki67_index = pos_count / total_cells

    # Classification logic: if more positive cells than negative, malignant
    if pos_count > neg_count:
        return 'malignant', ki67_index
    else:
        return 'benign', ki67_index

def get_ground_truth_malignancy(annotation_dir, image_name):
    """
    Get ground truth malignancy from H5 files

    Args:
        annotation_dir: Path to annotations directory
        image_name: Image filename (e.g., '0.png')

    Returns:
        'malignant' or 'benign' based on ground truth cell counts
    """
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

    pos_count = len(pos_points)
    neg_count = len(neg_points)

    # Ground truth classification
    if pos_count > neg_count:
        return 'malignant'
    else:
        return 'benign'

# ============================================================================
# MALIGNANCY CLASSIFICATION VALIDATION
# ============================================================================

def validate_malignancy_classification(checkpoint_path, data_path='/content/BCData', batch_size=8):
    """
    Validate malignancy classification accuracy on test set
    Returns classification metrics and confusion matrix
    """

    print(f"\n{'='*80}")
    print("🔬 MALIGNANCY CLASSIFICATION VALIDATION")
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

    # Create test dataset
    heatmap_gen = ImprovedPointHeatmapGenerator(sigma=8.0)

    try:
        test_dataset = Ki67PointDataset(
            image_dir=os.path.join(data_path, 'images/test'),
            annotation_dir=os.path.join(data_path, 'annotations/test'),
            transform=get_validation_augmentation(),
            heatmap_generator=heatmap_gen
        )
        print(f"✓ Test dataset created: {len(test_dataset)} images")
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return None

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Classification results
    predictions = []
    ground_truths = []
    ki67_indices = []
    cell_counts = []

    print("\nRunning malignancy classification...")
    print("Classifying each image as MALIGNANT or BENIGN...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing images"):
            images, heatmaps, names = batch
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            pred_heatmaps = torch.sigmoid(outputs)

            # Process each image in batch
            for i in range(len(names)):
                pred_pos = pred_heatmaps[i, 0].float().numpy()
                pred_neg = pred_heatmaps[i, 1].float().numpy()

                # Detect peaks (cell locations)
                pred_peaks_pos = peak_local_max(pred_pos, threshold_abs=0.3, min_distance=10)
                pred_peaks_neg = peak_local_max(pred_neg, threshold_abs=0.3, min_distance=10)

                pos_count = len(pred_peaks_pos)
                neg_count = len(pred_peaks_neg)

                # Classify malignancy
                prediction, ki67_index = classify_malignancy_from_cells(pos_count, neg_count)

                # Get ground truth
                ground_truth = get_ground_truth_malignancy(
                    os.path.join(data_path, 'annotations/test'),
                    names[i]
                )

                # Store results
                predictions.append(prediction)
                ground_truths.append(ground_truth)
                ki67_indices.append(ki67_index)
                cell_counts.append((pos_count, neg_count))

    # Convert to numpy arrays
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    ki67_indices = np.array(ki67_indices)

    # Calculate metrics
    accuracy = accuracy_score(ground_truths, predictions)

    # Confusion matrix
    cm = confusion_matrix(ground_truths, predictions, labels=['benign', 'malignant'])

    # Classification report
    report = classification_report(ground_truths, predictions,
                                 labels=['benign', 'malignant'],
                                 target_names=['Benign', 'Malignant'],
                                 output_dict=True)

    # Results
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': predictions,
        'ground_truths': ground_truths,
        'ki67_indices': ki67_indices,
        'cell_counts': cell_counts,
        'num_images': len(predictions)
    }

    # Print results
    print(f"\n{'='*80}")
    print("📊 MALIGNANCY CLASSIFICATION RESULTS")
    print(f"{'='*80}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Total Images: {len(predictions)}")
    print()

    print("CONFUSION MATRIX:")
    print("                 Predicted")
    print("                 Benign    Malignant")
    print(f"Actual  Benign    {cm[0,0]:<8}  {cm[0,1]:<8}")
    print(f"        Malignant {cm[1,0]:<8}  {cm[1,1]:<8}")
    print()

    print("CLASSIFICATION REPORT:")
    print(f"Benign    - Precision: {report['Benign']['precision']:.3f}, Recall: {report['Benign']['recall']:.3f}, F1: {report['Benign']['f1-score']:.3f}")
    print(f"Malignant - Precision: {report['Malignant']['precision']:.3f}, Recall: {report['Malignant']['recall']:.3f}, F1: {report['Malignant']['f1-score']:.3f}")
    print()

    # Statistics
    malignant_pred = np.sum(predictions == 'malignant')
    benign_pred = np.sum(predictions == 'benign')
    malignant_gt = np.sum(ground_truths == 'malignant')
    benign_gt = np.sum(ground_truths == 'benign')

    print("STATISTICS:")
    print(f"Ground Truth - Malignant: {malignant_gt}, Benign: {benign_gt}")
    print(f"Predictions  - Malignant: {malignant_pred}, Benign: {benign_pred}")
    print()

    # Ki-67 Index statistics
    malignant_ki67 = ki67_indices[ground_truths == 'malignant']
    benign_ki67 = ki67_indices[ground_truths == 'benign']

    print("KI-67 INDEX STATISTICS:")
    print(".3f")
    print(".3f")
    print()

    # Accuracy interpretation
    print("ACCURACY INTERPRETATION:")
    if accuracy >= 0.90:
        print("🎉 EXCELLENT: 90%+ classification accuracy!")
    elif accuracy >= 0.80:
        print("✅ VERY GOOD: 80%+ classification accuracy")
    elif accuracy >= 0.70:
        print("⚠️  GOOD: 70%+ classification accuracy")
    elif accuracy >= 0.60:
        print("🤔 FAIR: 60%+ classification accuracy - needs improvement")
    else:
        print("❌ POOR: <60% classification accuracy - retrain needed")
    print(f"{'='*80}\n")

    return results

# ============================================================================
# VISUALIZE CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(results):
    """Plot confusion matrix heatmap"""
    cm = results['confusion_matrix']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('Malignancy Classification Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('Actual', fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("🔬 Ki-67 Malignancy Classification Validation")
    print("="*80)

    # Configuration
    CHECKPOINT_PATH = '/content/ki67-point-epoch=68-val_peak_f1_avg=0.8503.ckpt'
    DATA_PATH = '/content/BCData'
    BATCH_SIZE = 8

    # Run validation
    results = validate_malignancy_classification(
        checkpoint_path=CHECKPOINT_PATH,
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE
    )

    if results:
        print("✅ Classification validation completed successfully!")
        print(f"📊 Key Results:")
        print(".1f")
        print(f"   Total Images: {results['num_images']}")

        # Plot confusion matrix
        print("\n📈 Generating confusion matrix visualization...")
        try:
            plot_confusion_matrix(results)
            print("✓ Confusion matrix plotted")
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")

    else:
        print("❌ Classification validation failed - check paths and files")

    print(f"\n{'='*80}")
    print("🎉 Script Complete!")
    print(f"{'='*80}")