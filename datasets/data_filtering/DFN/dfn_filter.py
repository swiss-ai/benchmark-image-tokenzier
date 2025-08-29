#!/usr/bin/env python3
"""
DFN (Data Filtering Networks) for Dataset Quality Scoring
Uses CLIP-based image-text alignment scores for data filtering
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union, Dict
import numpy as np
from PIL import Image
import open_clip
from tqdm import tqdm
import json


class DFNFilter:
    """
    DFN-based data filtering using CLIP image-text alignment scores.
    
    The filtering approach uses a CLIP model to compute alignment scores
    between images and their captions, which serves as a quality metric.
    Higher scores indicate better image-text alignment.
    """
    
    def __init__(
        self,
        model_name: str = 'hf-hub:apple/DFN-public',
        device: str = 'cuda',
        batch_size: int = 32
    ):
        """
        Initialize DFN filtering model
        
        Args:
            model_name: HuggingFace model hub identifier
                       Options: 'hf-hub:apple/DFN-public' (ViT-B/32)
            device: Device to run model on
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        
        # Load model and preprocessor
        print(f"Loading DFN filtering model: {model_name}")
        
        # Check if using transformers-based model (DFN-public)
        if 'DFN-public' in model_name:
            from transformers import CLIPModel, CLIPTokenizer
            from torchvision import transforms
            
            # Remove 'hf-hub:' prefix if present
            model_id = model_name.replace('hf-hub:', '')
            
            self.model = CLIPModel.from_pretrained(model_id)
            self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
            
            # Create manual preprocessor for DFN-public
            vision_config = self.model.config.vision_config
            image_size = vision_config.image_size
            self.preprocess = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
                    std=[0.26862954, 0.26130258, 0.27577711]
                ),
            ])
            self.use_transformers = True
        else:
            # Use open_clip for other models
            self.model, self.preprocess = open_clip.create_model_from_pretrained(model_name)
            
            # Get appropriate tokenizer based on model architecture
            if 'ViT-B-32' in model_name:
                self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
            elif 'ViT-H-14' in model_name:
                self.tokenizer = open_clip.get_tokenizer('ViT-H-14')
            elif 'ViT-B-16' in model_name:
                self.tokenizer = open_clip.get_tokenizer('ViT-B-16')
            else:
                # Default tokenizer
                self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
            self.use_transformers = False
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model architecture: {self.model.__class__.__name__}")
        
    def compute_filtering_score(
        self,
        image: Union[Image.Image, torch.Tensor],
        text: str
    ) -> float:
        """
        Compute DFN filtering score for a single image-text pair.
        
        The score represents the cosine similarity between image and text embeddings,
        scaled by the model's learned temperature parameter.
        
        Args:
            image: PIL Image or preprocessed tensor
            text: Caption or description text
            
        Returns:
            Filtering score (higher is better quality)
        """
        # Preprocess image if needed
        if isinstance(image, Image.Image):
            image_tensor = self.preprocess(image).unsqueeze(0)
        else:
            image_tensor = image.unsqueeze(0) if image.dim() == 3 else image
            
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            if self.use_transformers:
                # Use transformers API
                inputs = self.tokenizer([text], return_tensors="pt", padding=True).to(self.device)
                image_features = self.model.get_image_features(pixel_values=image_tensor)
                text_features = self.model.get_text_features(input_ids=inputs["input_ids"])
            else:
                # Use open_clip API
                text_tokens = self.tokenizer([text]).to(self.device)
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tokens)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            similarity = (image_features @ text_features.T).squeeze()
            
            # Apply temperature scaling if available
            if hasattr(self.model, 'logit_scale'):
                logit_scale = self.model.logit_scale.exp()
                similarity = similarity * logit_scale
                
        return float(similarity.cpu())
    
    def batch_compute_scores(
        self,
        image_text_pairs: List[Tuple[Union[Image.Image, torch.Tensor], str]],
        show_progress: bool = True
    ) -> List[float]:
        """
        Compute filtering scores for multiple image-text pairs efficiently.
        
        Args:
            image_text_pairs: List of (image, text) tuples
            show_progress: Show progress bar
            
        Returns:
            List of filtering scores
        """
        scores = []
        
        # Process in batches
        num_batches = (len(image_text_pairs) + self.batch_size - 1) // self.batch_size
        
        iterator = range(0, len(image_text_pairs), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Computing DFN scores", total=num_batches)
        
        for i in iterator:
            batch = image_text_pairs[i:i + self.batch_size]
            
            # Prepare batch tensors
            images = []
            texts = []
            for img, txt in batch:
                if isinstance(img, Image.Image):
                    images.append(self.preprocess(img))
                else:
                    images.append(img)
                texts.append(txt)
            
            # Stack images
            image_batch = torch.stack(images).to(self.device)
            
            with torch.no_grad():
                if self.use_transformers:
                    # Use transformers API
                    inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
                    image_features = self.model.get_image_features(pixel_values=image_batch)
                    text_features = self.model.get_text_features(input_ids=inputs["input_ids"])
                else:
                    # Use open_clip API
                    text_tokens = self.tokenizer(texts).to(self.device)
                    image_features = self.model.encode_image(image_batch)
                    text_features = self.model.encode_text(text_tokens)
                
                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Compute pairwise similarities (diagonal elements for matching pairs)
                similarities = image_features @ text_features.T
                
                # Apply temperature scaling if available
                if hasattr(self.model, 'logit_scale'):
                    logit_scale = self.model.logit_scale.exp()
                    similarities = similarities * logit_scale
                
                # Extract diagonal (matching pairs)
                batch_scores = torch.diagonal(similarities)
                
            scores.extend(batch_scores.cpu().tolist())
                
        return scores
    
    def compute_statistics(self, scores: List[float]) -> Dict[str, float]:
        """
        Compute statistics for filtering scores
        
        Args:
            scores: List of filtering scores
            
        Returns:
            Dictionary with statistics
        """
        scores_array = np.array(scores)
        return {
            "mean": float(np.mean(scores_array)),
            "std": float(np.std(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "median": float(np.median(scores_array)),
            "q25": float(np.percentile(scores_array, 25)),
            "q75": float(np.percentile(scores_array, 75)),
        }
    
    def filter_by_threshold(
        self,
        scores: List[float],
        threshold: Optional[float] = None,
        percentile: Optional[float] = None
    ) -> List[bool]:
        """
        Filter data based on score threshold or percentile
        
        Args:
            scores: List of filtering scores
            threshold: Absolute threshold (use scores >= threshold)
            percentile: Percentile threshold (e.g., 50 to keep top 50%)
            
        Returns:
            List of boolean masks indicating which samples to keep
        """
        scores_array = np.array(scores)
        
        if percentile is not None:
            threshold = np.percentile(scores_array, 100 - percentile)
        elif threshold is None:
            # Default: keep top 50%
            threshold = np.median(scores_array)
            
        return (scores_array >= threshold).tolist()
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "model_type": self.model.__class__.__name__,
            "vision_encoder": self.model.visual.__class__.__name__ if hasattr(self.model, 'visual') else "Unknown",
            "text_encoder": self.model.text.__class__.__name__ if hasattr(self.model, 'text') else "Unknown",
            "filtering_method": "CLIP image-text alignment score"
        }