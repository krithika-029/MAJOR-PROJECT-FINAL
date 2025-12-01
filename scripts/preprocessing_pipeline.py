"""
Ki-67 Breast Cancer Dataset Preprocessing Pipeline
===================================================
Comprehensive preprocessing system for whole slide images (WSI) to prepare
training data for U-Net with EfficientNet-B3 encoder.

Key Features:
- WSI format validation (.svs, .tiff, .ndpi, .jpeg, .png)
- Automated tissue detection using Otsu thresholding
- Macenko stain normalization
- Systematic patch extraction (256×256 with 50% overlap)
- Quality filtering (background/artifact removal)
- Feature extraction (statistical, texture, morphological)
- Heatmap generation with 2D Gaussian kernels
- Automated dataset splitting (train/val/test)
"""

import os
import numpy as np
import cv2
from PIL import Image
import h5py
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try to import openslide for WSI support
try:
    import openslide
    WSI_SUPPORT = True
except ImportError:
    print("⚠️  OpenSlide not installed. WSI formats (.svs, .ndpi) will not be supported.")
    print("   Install with: pip install openslide-python")
    WSI_SUPPORT = False


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    # Patch extraction
    patch_size: int = 256
    patch_overlap: float = 0.5  # 50% overlap
    
    # Quality thresholds
    min_tissue_percentage: float = 0.3  # 30% minimum tissue in patch
    blur_threshold: float = 100.0  # Laplacian variance threshold
    
    # Stain normalization
    enable_stain_norm: bool = True
    target_concentrations: np.ndarray = None  # Macenko reference
    
    # Gaussian kernel for heatmap
    gaussian_sigma: float = 3.0
    kernel_size: int = 25  # Must be odd
    
    # Output format
    output_patch_size: int = 640  # Final size after preprocessing
    save_visualization: bool = True
    

class MacenkoStainNormalizer:
    """
    Macenko stain normalization to standardize H&E staining variations
    Reference: Macenko et al., 2009, IEEE ISBI
    """
    
    def __init__(self):
        self.target_stains = None
        self.target_concentrations = None
        self.maxC_target = None
        
    def fit(self, target_image: np.ndarray):
        """Fit normalizer to target image"""
        stains, concentrations = self._get_stain_matrix(target_image)
        self.target_stains = stains
        self.target_concentrations = concentrations
        self.maxC_target = np.percentile(concentrations, 99, axis=1).reshape(-1, 1)
        
    def transform(self, image: np.ndarray) -> np.ndarray:
        """Normalize input image to target appearance"""
        if self.target_stains is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        # Get stain matrix for input image
        stains, concentrations = self._get_stain_matrix(image)
        
        # Normalize concentrations
        maxC_source = np.percentile(concentrations, 99, axis=1).reshape(-1, 1)
        concentrations *= (self.maxC_target / maxC_source)
        
        # Reconstruct image using target stains
        normalized = 255 * np.exp(-self.target_stains @ concentrations)
        normalized = normalized.T.reshape(image.shape).astype(np.uint8)
        
        return normalized
    
    def _get_stain_matrix(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract stain matrix using PCA"""
        # Convert to optical density
        image = image.astype(np.float64) + 1
        OD = -np.log(image / 255)
        OD = OD.reshape(-1, 3)
        
        # Remove transparent pixels
        OD = OD[~np.any(OD < 0.15, axis=1)]
        
        # Compute eigenvectors
        _, eigvecs = np.linalg.eigh(np.cov(OD.T))
        eigvecs = eigvecs[:, [2, 1]]  # Select top 2
        
        # Project to plane
        proj = OD @ eigvecs
        
        # Find robust extreme angles
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        min_phi = np.percentile(phi, 1)
        max_phi = np.percentile(phi, 99)
        
        # Extract stain vectors
        v1 = eigvecs @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v2 = eigvecs @ np.array([np.cos(max_phi), np.sin(max_phi)])
        
        # Normalize
        if v1[0] > v2[0]:
            HE = np.array([v1, v2])
        else:
            HE = np.array([v2, v1])
        
        # Compute concentrations
        Y = OD @ np.linalg.pinv(HE)
        
        return HE, Y.T


class TissueDetector:
    """Automated tissue detection using Otsu thresholding"""
    
    @staticmethod
    def detect_tissue_mask(image: np.ndarray, blur_kernel: int = 5) -> np.ndarray:
        """
        Detect tissue regions in image
        
        Args:
            image: RGB image
            blur_kernel: Gaussian blur kernel size
            
        Returns:
            Binary mask where 1 = tissue, 0 = background
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        
        # Otsu thresholding
        _, tissue_mask = cv2.threshold(blurred, 0, 255, 
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        return tissue_mask


class QualityFilter:
    """Filter patches based on quality metrics"""
    
    @staticmethod
    def calculate_blur_score(image: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    @staticmethod
    def calculate_tissue_percentage(image: np.ndarray, tissue_mask: np.ndarray) -> float:
        """Calculate percentage of tissue in patch"""
        return np.sum(tissue_mask > 0) / tissue_mask.size
    
    @staticmethod
    def is_valid_patch(image: np.ndarray, tissue_mask: np.ndarray, 
                       config: PreprocessingConfig) -> bool:
        """Check if patch meets quality criteria"""
        # Check tissue percentage
        tissue_pct = QualityFilter.calculate_tissue_percentage(image, tissue_mask)
        if tissue_pct < config.min_tissue_percentage:
            return False
        
        # Check blur
        blur_score = QualityFilter.calculate_blur_score(image)
        if blur_score < config.blur_threshold:
            return False
        
        return True


class FeatureExtractor:
    """Extract statistical, texture, and morphological features"""
    
    @staticmethod
    def extract_statistical_features(image: np.ndarray) -> Dict[str, float]:
        """Extract statistical measures"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        return {
            'mean': np.mean(gray),
            'std': np.std(gray),
            'variance': np.var(gray),
            'skewness': float(np.mean(((gray - np.mean(gray)) / np.std(gray)) ** 3)),
            'kurtosis': float(np.mean(((gray - np.mean(gray)) / np.std(gray)) ** 4))
        }
    
    @staticmethod
    def extract_glcm_features(image: np.ndarray) -> Dict[str, float]:
        """Extract Gray-Level Co-occurrence Matrix texture features"""
        from skimage.feature import graycomatrix, graycoprops
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Compute GLCM
        glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=256, symmetric=True, normed=True)
        
        # Extract properties
        return {
            'glcm_contrast': float(np.mean(graycoprops(glcm, 'contrast'))),
            'glcm_dissimilarity': float(np.mean(graycoprops(glcm, 'dissimilarity'))),
            'glcm_homogeneity': float(np.mean(graycoprops(glcm, 'homogeneity'))),
            'glcm_energy': float(np.mean(graycoprops(glcm, 'energy'))),
            'glcm_correlation': float(np.mean(graycoprops(glcm, 'correlation')))
        }
    
    @staticmethod
    def extract_morphological_features(image: np.ndarray) -> Dict[str, float]:
        """Extract morphological features describing nuclear characteristics"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect nuclei using adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return {'nucleus_count': 0, 'avg_nucleus_area': 0, 'avg_nucleus_perimeter': 0}
        
        areas = [cv2.contourArea(c) for c in contours]
        perimeters = [cv2.arcLength(c, True) for c in contours]
        
        return {
            'nucleus_count': len(contours),
            'avg_nucleus_area': np.mean(areas),
            'avg_nucleus_perimeter': np.mean(perimeters),
            'nucleus_density': len(contours) / (image.shape[0] * image.shape[1])
        }


class HeatmapGenerator:
    """Generate Gaussian heatmaps from point annotations"""
    
    def __init__(self, sigma: float = 3.0, kernel_size: int = 25):
        """
        Args:
            sigma: Standard deviation of Gaussian kernel
            kernel_size: Size of kernel (must be odd)
        """
        self.sigma = sigma
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # Pre-compute Gaussian kernel
        self.kernel = self._create_gaussian_kernel()
    
    def _create_gaussian_kernel(self) -> np.ndarray:
        """Create 2D Gaussian kernel"""
        ax = np.arange(-self.kernel_size // 2 + 1., self.kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        
        kernel = np.exp(-(xx**2 + yy**2) / (2. * self.sigma**2))
        return kernel / kernel.max()
    
    def generate_heatmap(self, points: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Generate heatmap from point annotations
        
        Args:
            points: Nx2 array of (x, y) coordinates
            image_shape: (height, width) of output heatmap
            
        Returns:
            Heatmap with values in [0, 1]
        """
        heatmap = np.zeros(image_shape, dtype=np.float32)
        
        if len(points) == 0:
            return heatmap
        
        half_kernel = self.kernel_size // 2
        
        for x, y in points:
            x, y = int(x), int(y)
            
            # Compute bounds
            x_min = max(0, x - half_kernel)
            x_max = min(image_shape[1], x + half_kernel + 1)
            y_min = max(0, y - half_kernel)
            y_max = min(image_shape[0], y + half_kernel + 1)
            
            # Compute kernel bounds
            kx_min = half_kernel - (x - x_min)
            kx_max = half_kernel + (x_max - x)
            ky_min = half_kernel - (y - y_min)
            ky_max = half_kernel + (y_max - y)
            
            # Add Gaussian to heatmap (use maximum to handle overlaps)
            heatmap[y_min:y_max, x_min:x_max] = np.maximum(
                heatmap[y_min:y_max, x_min:x_max],
                self.kernel[ky_min:ky_max, kx_min:kx_max]
            )
        
        return heatmap


class WSIPreprocessor:
    """Main preprocessing pipeline for WSI to training dataset"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.stain_normalizer = MacenkoStainNormalizer()
        self.tissue_detector = TissueDetector()
        self.quality_filter = QualityFilter()
        self.feature_extractor = FeatureExtractor()
        self.heatmap_generator = HeatmapGenerator(
            sigma=config.gaussian_sigma,
            kernel_size=config.kernel_size
        )
        
    def fit_stain_normalizer(self, reference_image_path: str):
        """Fit stain normalizer to reference image"""
        img = np.array(Image.open(reference_image_path))
        self.stain_normalizer.fit(img)
        print(f"✅ Stain normalizer fitted to: {reference_image_path}")
    
    def load_image(self, image_path: str, level: int = 0) -> np.ndarray:
        """
        Load image from various formats
        
        Args:
            image_path: Path to image file
            level: Pyramid level for WSI (0 = highest resolution)
            
        Returns:
            RGB image as numpy array
        """
        image_path = Path(image_path)
        
        # Check if WSI format
        if image_path.suffix.lower() in ['.svs', '.ndpi', '.tiff', '.tif']:
            if not WSI_SUPPORT:
                raise RuntimeError("OpenSlide not installed. Cannot process WSI formats.")
            
            slide = openslide.OpenSlide(str(image_path))
            image = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            slide.close()
        else:
            # Standard image formats
            image = np.array(Image.open(image_path))
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        return image
    
    def extract_patches(self, image: np.ndarray, tissue_mask: np.ndarray) -> List[Dict]:
        """
        Extract patches with overlap from image
        
        Returns:
            List of patch dictionaries containing image, mask, position
        """
        h, w = image.shape[:2]
        patch_size = self.config.patch_size
        stride = int(patch_size * (1 - self.config.patch_overlap))
        
        patches = []
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                # Extract patch
                patch_img = image[y:y+patch_size, x:x+patch_size]
                patch_mask = tissue_mask[y:y+patch_size, x:x+patch_size]
                
                # Quality filtering
                if not self.quality_filter.is_valid_patch(patch_img, patch_mask, self.config):
                    continue
                
                patches.append({
                    'image': patch_img,
                    'mask': patch_mask,
                    'position': (x, y),
                    'blur_score': self.quality_filter.calculate_blur_score(patch_img),
                    'tissue_percentage': self.quality_filter.calculate_tissue_percentage(patch_img, patch_mask)
                })
        
        return patches
    
    def load_annotations(self, annotation_path: str) -> Dict[str, np.ndarray]:
        """
        Load point annotations from H5 file
        
        Expected format:
        - positive.h5: contains 'points' dataset with positive cell coordinates
        - negative.h5: contains 'points' dataset with negative cell coordinates
        
        Returns:
            Dictionary with 'positive' and 'negative' point arrays
        """
        annotation_path = Path(annotation_path)
        annotations = {}
        
        for cell_type in ['positive', 'negative']:
            h5_file = annotation_path / f"{cell_type}.h5"
            if h5_file.exists():
                with h5py.File(h5_file, 'r') as f:
                    if 'points' in f:
                        annotations[cell_type] = np.array(f['points'])
                    else:
                        annotations[cell_type] = np.array([])
            else:
                annotations[cell_type] = np.array([])
        
        return annotations
    
    def process_image(self, image_path: str, annotation_path: Optional[str] = None,
                     output_dir: str = None) -> Dict:
        """
        Complete preprocessing pipeline for single image
        
        Args:
            image_path: Path to input image
            annotation_path: Path to directory containing annotation H5 files
            output_dir: Directory to save processed patches
            
        Returns:
            Processing statistics and metadata
        """
        print(f"\n📄 Processing: {Path(image_path).name}")
        
        # Load image
        image = self.load_image(image_path)
        print(f"   Image size: {image.shape}")
        
        # Detect tissue
        tissue_mask = self.tissue_detector.detect_tissue_mask(image)
        tissue_area = np.sum(tissue_mask > 0) / tissue_mask.size * 100
        print(f"   Tissue area: {tissue_area:.2f}%")
        
        # Stain normalization
        if self.config.enable_stain_norm and self.stain_normalizer.target_stains is not None:
            image = self.stain_normalizer.transform(image)
            print("   ✓ Stain normalized")
        
        # Extract patches
        patches = self.extract_patches(image, tissue_mask)
        print(f"   Extracted {len(patches)} valid patches")
        
        # Load annotations if available
        annotations = None
        if annotation_path:
            annotations = self.load_annotations(annotation_path)
            pos_count = len(annotations.get('positive', []))
            neg_count = len(annotations.get('negative', []))
            print(f"   Annotations: {pos_count} positive, {neg_count} negative cells")
        
        # Save patches
        saved_patches = []
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            image_name = Path(image_path).stem
            
            for idx, patch_data in enumerate(tqdm(patches, desc="Saving patches")):
                patch_name = f"{image_name}_patch_{idx:04d}"
                patch_dir = output_dir / patch_name
                patch_dir.mkdir(exist_ok=True)
                
                # Save patch image
                Image.fromarray(patch_data['image']).save(patch_dir / 'image.png')
                
                # Save tissue mask
                Image.fromarray(patch_data['mask']).save(patch_dir / 'tissue_mask.png')
                
                # Generate and save heatmaps if annotations available
                if annotations:
                    x_offset, y_offset = patch_data['position']
                    patch_size = self.config.patch_size
                    
                    for cell_type in ['positive', 'negative']:
                        points = annotations[cell_type]
                        if len(points) > 0:
                            # Filter points within patch
                            mask = ((points[:, 0] >= x_offset) & (points[:, 0] < x_offset + patch_size) &
                                   (points[:, 1] >= y_offset) & (points[:, 1] < y_offset + patch_size))
                            patch_points = points[mask] - np.array([x_offset, y_offset])
                            
                            # Generate heatmap
                            heatmap = self.heatmap_generator.generate_heatmap(
                                patch_points, (patch_size, patch_size)
                            )
                            
                            # Save as H5 for training
                            with h5py.File(patch_dir / f'{cell_type}_heatmap.h5', 'w') as f:
                                f.create_dataset('heatmap', data=heatmap, compression='gzip')
                                f.create_dataset('points', data=patch_points)
                                f.attrs['cell_count'] = len(patch_points)
                
                # Extract and save features
                features = {
                    **self.feature_extractor.extract_statistical_features(patch_data['image']),
                    **self.feature_extractor.extract_glcm_features(patch_data['image']),
                    **self.feature_extractor.extract_morphological_features(patch_data['image']),
                    'blur_score': patch_data['blur_score'],
                    'tissue_percentage': patch_data['tissue_percentage']
                }
                
                with open(patch_dir / 'features.json', 'w') as f:
                    json.dump(features, f, indent=2)
                
                saved_patches.append({
                    'patch_name': patch_name,
                    'position': patch_data['position'],
                    'features': features
                })
        
        # Return processing summary
        return {
            'image_path': str(image_path),
            'image_shape': image.shape,
            'tissue_percentage': tissue_area,
            'num_patches': len(patches),
            'num_saved': len(saved_patches),
            'has_annotations': annotations is not None,
            'timestamp': datetime.now().isoformat()
        }


def create_dataset_split(dataset_dir: str, train_ratio: float = 0.7,
                        val_ratio: float = 0.15, test_ratio: float = 0.15,
                        seed: int = 42):
    """
    Split dataset into train/val/test sets
    
    Args:
        dataset_dir: Directory containing processed patches
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    """
    dataset_dir = Path(dataset_dir)
    
    # Get all patch directories
    patch_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    
    # Shuffle
    np.random.seed(seed)
    np.random.shuffle(patch_dirs)
    
    # Split
    n = len(patch_dirs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    splits = {
        'train': patch_dirs[:train_end],
        'val': patch_dirs[train_end:val_end],
        'test': patch_dirs[val_end:]
    }
    
    # Create split directories and move patches
    for split_name, patches in splits.items():
        split_dir = dataset_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        print(f"\n{split_name.upper()}: {len(patches)} patches")
        
        for patch_dir in tqdm(patches, desc=f"Moving {split_name}"):
            target_dir = split_dir / patch_dir.name
            if not target_dir.exists():
                patch_dir.rename(target_dir)
    
    # Save split metadata
    split_info = {
        'train': len(splits['train']),
        'val': len(splits['val']),
        'test': len(splits['test']),
        'ratios': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio},
        'seed': seed,
        'created': datetime.now().isoformat()
    }
    
    with open(dataset_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✅ Dataset split complete!")
    print(f"   Train: {split_info['train']} ({train_ratio*100}%)")
    print(f"   Val:   {split_info['val']} ({val_ratio*100}%)")
    print(f"   Test:  {split_info['test']} ({test_ratio*100}%)")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    
    print("=" * 80)
    print("Ki-67 Dataset Preprocessing Pipeline")
    print("=" * 80)
    
    # Configuration
    config = PreprocessingConfig(
        patch_size=256,
        patch_overlap=0.5,
        min_tissue_percentage=0.3,
        blur_threshold=100.0,
        enable_stain_norm=True,
        gaussian_sigma=3.0,
        kernel_size=25
    )
    
    # Initialize preprocessor
    preprocessor = WSIPreprocessor(config)
    
    # STEP 1: Fit stain normalizer (use a reference H&E image)
    # reference_image = "path/to/reference_he_image.png"
    # preprocessor.fit_stain_normalizer(reference_image)
    
    # STEP 2: Process images
    input_dir = Path("raw_images")  # Directory with your WSI/images
    annotation_dir = Path("annotations")  # Directory with annotation H5 files
    output_dir = Path("processed_dataset")
    
    # Example: Process all images in directory
    if input_dir.exists():
        image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + \
                     list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.tiff"))
        
        print(f"\nFound {len(image_files)} images to process")
        
        processing_log = []
        
        for image_path in image_files:
            # Check for annotations
            image_name = image_path.stem
            annot_path = annotation_dir / image_name
            
            if not annot_path.exists():
                print(f"⚠️  No annotations found for {image_name}, skipping...")
                continue
            
            # Process image
            result = preprocessor.process_image(
                str(image_path),
                str(annot_path),
                str(output_dir)
            )
            
            processing_log.append(result)
        
        # Save processing log
        with open(output_dir / 'processing_log.json', 'w') as f:
            json.dump(processing_log, f, indent=2)
        
        print(f"\n✅ Preprocessing complete! {len(processing_log)} images processed")
        
        # STEP 3: Create dataset splits
        create_dataset_split(str(output_dir), train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    else:
        print(f"\n⚠️  Input directory not found: {input_dir}")
        print("Please create the directory structure:")
        print("  raw_images/")
        print("    ├── image1.png")
        print("    └── image2.png")
        print("  annotations/")
        print("    ├── image1/")
        print("    │   ├── positive.h5")
        print("    │   └── negative.h5")
        print("    └── image2/")
        print("        ├── positive.h5")
        print("        └── negative.h5")
