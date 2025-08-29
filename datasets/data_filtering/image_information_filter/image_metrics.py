"""
Image Information Metrics for Dataset Quality Assessment

This module provides efficient implementations of four key image information metrics:
1. Luminance Entropy (normalized)
2. Spatial Information (SI - ITU-T P.910)
3. Edge Density (Canny-based)
4. Variance of Laplacian (focus/sharpness)
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass
from enum import Enum


class ColorSpace(Enum):
    """Supported color space conversion standards"""
    BT601 = "bt601"  # SDTV standard
    BT709 = "bt709"  # HDTV standard


@dataclass
class MetricResult:
    """Container for metric results with metadata"""
    value: float
    metric_name: str
    original_shape: Tuple[int, int]
    processing_shape: Optional[Tuple[int, int]] = None
    metadata: Optional[Dict[str, Any]] = None


class ImageMetrics:
    """
    Efficient implementation of image information metrics for large-scale dataset evaluation.
    
    Metrics can be computed at original resolution with automatic normalization for
    resolution-independent comparison, or at a fixed reference size.
    """
    
    def __init__(self, 
                 reference_size: Optional[Tuple[int, int]] = None,
                 color_space: ColorSpace = ColorSpace.BT709,
                 use_gpu: bool = False,
                 resolution_independent: bool = False):
        """
        Initialize the ImageMetrics calculator.
        
        Args:
            reference_size: Target size (width, height) for normalization. 
                          If None, metrics computed at original resolution.
            color_space: Color space standard for RGB to grayscale conversion
            use_gpu: Whether to use GPU acceleration (requires opencv-contrib-python)
            resolution_independent: If True and reference_size is None, apply
                                  resolution normalization to metrics
        """
        self.reference_size = reference_size
        self.color_space = color_space
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.resolution_independent = resolution_independent
        
        # Color conversion matrices
        self.luma_weights = {
            ColorSpace.BT601: np.array([0.299, 0.587, 0.114]),
            ColorSpace.BT709: np.array([0.2126, 0.7152, 0.0722])
        }
        
    def _resize_if_needed(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Resize image to reference size if specified.
        
        Returns:
            Tuple of (processed_image, original_shape)
        """
        original_shape = image.shape[:2]
        
        if self.reference_size is not None:
            # Use INTER_LANCZOS4 for high-quality downsampling, INTER_CUBIC for upsampling
            if image.shape[1] > self.reference_size[0] or image.shape[0] > self.reference_size[1]:
                interpolation = cv2.INTER_LANCZOS4
            else:
                interpolation = cv2.INTER_CUBIC
                
            image = cv2.resize(image, self.reference_size, interpolation=interpolation)
            
        return image, original_shape
    
    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale using specified color space standard.
        
        Args:
            image: Input image (RGB or grayscale)
            
        Returns:
            Grayscale image as uint8
        """
        if len(image.shape) == 2:
            return image
        
        if len(image.shape) == 3:
            # Ensure we have 3 channels
            if image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]
            elif image.shape[2] != 3:
                raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
            
            # Apply color space conversion
            weights = self.luma_weights[self.color_space]
            grayscale = np.dot(image, weights)
            
            # Ensure uint8 output
            if grayscale.dtype != np.uint8:
                grayscale = np.clip(grayscale, 0, 255).astype(np.uint8)
                
            return grayscale
        
        raise ValueError(f"Invalid image shape: {image.shape}")
    
    def luminance_entropy(self, image: np.ndarray, normalize: bool = True) -> MetricResult:
        """
        Calculate normalized Shannon entropy of the luminance histogram.
        
        The entropy measures the randomness/variety in the grayscale intensity distribution.
        Higher values indicate more information content.
        
        Args:
            image: Input image (RGB or grayscale)
            normalize: If True, normalize by max entropy (8 bits for uint8)
            
        Returns:
            MetricResult with entropy value in [0, 1] if normalized
        """
        # Preprocessing
        image, original_shape = self._resize_if_needed(image)
        grayscale = self._to_grayscale(image)
        
        # Calculate histogram
        hist, _ = np.histogram(grayscale, bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        
        # Normalize to probability distribution
        hist = hist / hist.sum()
        
        # Calculate Shannon entropy
        # Only consider non-zero bins
        hist = hist[hist > 0]
        if len(hist) == 0:
            # Completely empty histogram
            entropy = 0.0
        else:
            entropy = -np.sum(hist * np.log2(hist))
        
        # Normalize by maximum possible entropy for 8-bit images
        if normalize:
            max_entropy = np.log2(256)  # 8 bits
            entropy = entropy / max_entropy
        
        return MetricResult(
            value=float(entropy),
            metric_name="luminance_entropy",
            original_shape=original_shape,
            processing_shape=grayscale.shape[:2],
            metadata={"normalized": normalize, "color_space": self.color_space.value}
        )
    
    def spatial_information(self, image: np.ndarray, blur_kernel: Optional[int] = None) -> MetricResult:
        """
        Calculate Spatial Information (SI) according to ITU-T P.910.
        
        SI measures the spatial complexity of an image using the standard deviation
        of the Sobel-filtered luminance.
        
        Args:
            image: Input image (RGB or grayscale)
            blur_kernel: Optional Gaussian blur kernel size for noise suppression
            
        Returns:
            MetricResult with SI value (typically 0-100+ for natural images)
        """
        # Preprocessing
        image, original_shape = self._resize_if_needed(image)
        grayscale = self._to_grayscale(image)
        
        # Convert to float for accurate gradient computation
        grayscale = grayscale.astype(np.float32)
        
        # Optional noise suppression
        if blur_kernel is not None and blur_kernel > 0:
            grayscale = cv2.GaussianBlur(grayscale, (blur_kernel, blur_kernel), 0)
        
        # Compute Sobel gradients
        sobel_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Exclude edge pixels as per ITU-T P.910 specification
        magnitude = magnitude[1:-1, 1:-1]
        
        # Calculate standard deviation (SI)
        si = np.std(magnitude)
        
        return MetricResult(
            value=float(si),
            metric_name="spatial_information",
            original_shape=original_shape,
            processing_shape=grayscale.shape[:2],
            metadata={"blur_kernel": blur_kernel, "color_space": self.color_space.value}
        )
    
    def edge_density(self, 
                    image: np.ndarray, 
                    method: str = "canny",
                    low_threshold: Optional[float] = None,
                    high_threshold: Optional[float] = None,
                    auto_threshold: bool = True) -> MetricResult:
        """
        Calculate edge density as the fraction of pixels classified as edges.
        
        Args:
            image: Input image (RGB or grayscale)
            method: Edge detection method ("canny" or "sobel")
            low_threshold: Lower threshold for Canny (ignored if auto_threshold=True)
            high_threshold: Upper threshold for Canny (ignored if auto_threshold=True)
            auto_threshold: If True, automatically determine thresholds using Otsu's method
            
        Returns:
            MetricResult with edge density value in [0, 1]
        """
        # Preprocessing
        image, original_shape = self._resize_if_needed(image)
        grayscale = self._to_grayscale(image)
        
        # Light Gaussian blur for noise suppression
        grayscale = cv2.GaussianBlur(grayscale, (3, 3), 0)
        
        if method.lower() == "canny":
            if auto_threshold:
                # Use Otsu's method to determine high threshold
                high_threshold, _ = cv2.threshold(grayscale, 0, 255, 
                                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                low_threshold = 0.5 * high_threshold
            else:
                # Use provided thresholds or defaults
                if high_threshold is None:
                    high_threshold = 100
                if low_threshold is None:
                    low_threshold = 50
            
            # Apply Canny edge detection
            edges = cv2.Canny(grayscale, low_threshold, high_threshold)
            
        elif method.lower() == "sobel":
            # Sobel magnitude thresholding
            sobel_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Threshold using Otsu on magnitude
            if auto_threshold:
                threshold = np.percentile(magnitude, 90)  # Top 10% as edges
            else:
                threshold = high_threshold if high_threshold is not None else 50
            
            edges = (magnitude > threshold).astype(np.uint8) * 255
            
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
        
        # Calculate edge density
        edge_density_value = np.count_nonzero(edges) / edges.size
        
        return MetricResult(
            value=float(edge_density_value),
            metric_name="edge_density",
            original_shape=original_shape,
            processing_shape=grayscale.shape[:2],
            metadata={
                "method": method,
                "auto_threshold": auto_threshold,
                "low_threshold": float(low_threshold) if low_threshold else None,
                "high_threshold": float(high_threshold) if high_threshold else None,
                "color_space": self.color_space.value
            }
        )
    
    def brenner_focus(self,
                     image: np.ndarray,
                     step: int = 2,
                     direction: str = 'both') -> MetricResult:
        """
        Calculate Brenner's focus measure for sharpness assessment.
        
        Brenner's method computes the squared gradient difference between pixels
        separated by 'step' pixels. Higher values indicate better focus.
        More sensitive to edges than Variance of Laplacian.
        
        Args:
            image: Input image (RGB or grayscale)
            step: Pixel step for gradient computation (default 2, as per original paper)
            direction: 'horizontal', 'vertical', or 'both'
            
        Returns:
            MetricResult with Brenner focus value
        """
        # Preprocessing
        image, original_shape = self._resize_if_needed(image)
        grayscale = self._to_grayscale(image).astype(np.float32)
        
        brenner_value = 0.0
        
        # Horizontal differences: I(x+step, y) - I(x, y)
        if direction in ['horizontal', 'both']:
            if grayscale.shape[1] > step:
                diff_h = grayscale[:, step:] - grayscale[:, :-step]
                brenner_value += np.sum(diff_h ** 2)
        
        # Vertical differences: I(x, y+step) - I(x, y)
        if direction in ['vertical', 'both']:
            if grayscale.shape[0] > step:
                diff_v = grayscale[step:, :] - grayscale[:-step, :]
                brenner_value += np.sum(diff_v ** 2)
        
        # Normalize appropriately based on settings
        if self.resolution_independent and self.reference_size is None:
            # For resolution independence at native resolution, use sqrt(area) normalization
            # This provides the best consistency across different image sizes
            image_area = grayscale.shape[0] * grayscale.shape[1]
            if image_area > 0:
                brenner_value = brenner_value / np.sqrt(image_area)
            normalization_method = 'sqrt_area'
        else:
            # Original normalization by number of pixel pairs (for fixed size comparison)
            if direction == 'both':
                n_pairs = (grayscale.shape[0] - step) * grayscale.shape[1] + \
                         grayscale.shape[0] * (grayscale.shape[1] - step)
            elif direction == 'horizontal':
                n_pairs = grayscale.shape[0] * (grayscale.shape[1] - step)
            else:  # vertical
                n_pairs = (grayscale.shape[0] - step) * grayscale.shape[1]
            
            if n_pairs > 0:
                brenner_value = brenner_value / n_pairs
            normalization_method = 'pixel_pairs'
        
        return MetricResult(
            value=float(brenner_value),
            metric_name="brenner_focus",
            original_shape=original_shape,
            metadata={
                'step': step,
                'direction': direction,
                'normalization': normalization_method,
                'resolution_independent': self.resolution_independent
            }
        )
    
    def variance_of_laplacian(self, 
                             image: np.ndarray,
                             kernel_size: int = 3,
                             normalize_by_area: bool = False) -> MetricResult:
        """
        Calculate Variance of Laplacian as a focus/sharpness measure.
        
        Higher variance indicates sharper, more in-focus images.
        
        Args:
            image: Input image (RGB or grayscale)
            kernel_size: Size of Laplacian kernel (1, 3, 5, or 7)
            normalize_by_area: If True, normalize by image area for resolution independence
            
        Returns:
            MetricResult with VoL value (typically 0-1000+ for natural images)
        """
        # Preprocessing
        image, original_shape = self._resize_if_needed(image)
        grayscale = self._to_grayscale(image)
        
        # Apply Laplacian operator
        # CV_64F is important to preserve negative values
        laplacian = cv2.Laplacian(grayscale, cv2.CV_64F, ksize=kernel_size)
        
        # Calculate variance
        variance = laplacian.var()
        
        # Normalize for kernel size to maintain consistent scale
        # Empirically determined factors for 512x512 images (default reference size)
        # These factors normalize all kernel sizes to similar scale as ksize=3
        kernel_normalization = {
            1: 4.0,      # ksize=1 produces ~0.25x variance, multiply by 4
            3: 1.0,      # baseline
            5: 1/13.0,   # ksize=5 produces ~13x variance, divide by 13
            7: 1/820.0   # ksize=7 produces ~820x variance, divide by 820
        }
        
        if kernel_size in kernel_normalization:
            variance = variance * kernel_normalization[kernel_size]
        else:
            warnings.warn(f"Unsupported VoL kernel size {kernel_size}, using no normalization")
        
        # Apply resolution normalization if needed
        if (self.resolution_independent and self.reference_size is None) or normalize_by_area:
            # Normalize by sqrt(area) for resolution independence
            area = grayscale.shape[0] * grayscale.shape[1]
            variance = variance / np.sqrt(area)
        
        return MetricResult(
            value=float(variance),
            metric_name="variance_of_laplacian",
            original_shape=original_shape,
            processing_shape=grayscale.shape[:2],
            metadata={
                "kernel_size": kernel_size,
                "normalize_by_area": normalize_by_area,
                "color_space": self.color_space.value
            }
        )
    
    def compute_all_metrics_normalized(self, image: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Compute all metrics with resolution-independent normalization.
        This is the recommended method for comparing images of different sizes.
        
        Args:
            image: Input image (RGB or grayscale)
            **kwargs: Additional parameters for individual metrics
            
        Returns:
            Dictionary mapping metric names to normalized float values
        """
        # Store original settings
        original_resolution_independent = self.resolution_independent
        original_reference_size = self.reference_size
        
        # Enable resolution-independent mode for native resolution evaluation
        self.resolution_independent = True
        self.reference_size = None
        
        results = {}
        
        try:
            # Brenner with sqrt(area) normalization
            brenner = self.brenner_focus(image)
            results['brenner_focus'] = brenner.value
            
            # VoL with sqrt(area) normalization
            vol = self.variance_of_laplacian(image, normalize_by_area=True)
            results['variance_of_laplacian'] = vol.value
            
            # SI is already normalized (standard deviation)
            si = self.spatial_information(image)
            results['spatial_information'] = si.value
            
            # Edge density is already normalized (fraction)
            edge = self.edge_density(image)
            results['edge_density'] = edge.value
            
            # Entropy is already normalized
            entropy = self.luminance_entropy(image)
            results['luminance_entropy'] = entropy.value
            
            # Add image dimensions for reference
            h, w = image.shape[:2] if len(image.shape) >= 2 else (0, 0)
            results['image_width'] = w
            results['image_height'] = h
            
        finally:
            # Restore original settings
            self.resolution_independent = original_resolution_independent
            self.reference_size = original_reference_size
        
        return results
    
    def compute_all_metrics(self, image: np.ndarray, **kwargs) -> Dict[str, MetricResult]:
        """
        Compute all four metrics for an image.
        
        Args:
            image: Input image (RGB or grayscale)
            **kwargs: Additional parameters for individual metrics
            
        Returns:
            Dictionary mapping metric names to MetricResult objects
        """
        results = {}
        
        # Extract metric-specific parameters
        entropy_params = {k.replace('entropy_', ''): v 
                         for k, v in kwargs.items() if k.startswith('entropy_')}
        si_params = {k.replace('si_', ''): v 
                    for k, v in kwargs.items() if k.startswith('si_')}
        edge_params = {k.replace('edge_', ''): v 
                      for k, v in kwargs.items() if k.startswith('edge_')}
        vol_params = {k.replace('vol_', ''): v 
                     for k, v in kwargs.items() if k.startswith('vol_')}
        
        # Compute metrics
        try:
            results['luminance_entropy'] = self.luminance_entropy(image, **entropy_params)
        except Exception as e:
            warnings.warn(f"Failed to compute luminance entropy: {e}")
            
        try:
            results['spatial_information'] = self.spatial_information(image, **si_params)
        except Exception as e:
            warnings.warn(f"Failed to compute spatial information: {e}")
            
        try:
            results['edge_density'] = self.edge_density(image, **edge_params)
        except Exception as e:
            warnings.warn(f"Failed to compute edge density: {e}")
            
        try:
            results['variance_of_laplacian'] = self.variance_of_laplacian(image, **vol_params)
        except Exception as e:
            warnings.warn(f"Failed to compute variance of Laplacian: {e}")
        
        try:
            results['brenner_focus'] = self.brenner_focus(image)
        except Exception as e:
            warnings.warn(f"Failed to compute Brenner focus: {e}")
        
        return results