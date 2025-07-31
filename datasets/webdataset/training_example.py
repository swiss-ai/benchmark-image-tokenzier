#!/usr/bin/env python3

"""
Example script showing how to use the distributed LLaVA WebDataset loader
for AI model training with PyTorch.

This demonstrates:
1. Distributed data loading
2. Efficient batching 
3. Image preprocessing
4. Text processing
5. Integration with training loops
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from distributed_webdataset_loader import DistributedLLaVAWebDataset
import argparse
from typing import Dict, Any


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        print("world_size: ", world_size)
        
        # Use appropriate backend based on device availability
        if torch.cuda.is_available():
            # Try NCCL first, fallback to GLOO for GPU if NCCL not available
            try:
                dist.init_process_group(backend='nccl')
                torch.cuda.set_device(local_rank)
                print(f"Using NCCL backend on GPU {local_rank}")
            except RuntimeError as e:
                if "NCCL" in str(e):
                    print(f"NCCL not available, using GLOO backend on GPU {local_rank}")
                    dist.init_process_group(backend='gloo')
                    torch.cuda.set_device(local_rank)  # Still use GPU with GLOO
                else:
                    raise e
        else:
            # CPU-only training
            print("CUDA not available, using GLOO backend for CPU training")
            dist.init_process_group(backend='gloo')
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def create_image_transform(image_size: int = 224):
    """Create image preprocessing transforms."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def create_training_dataloader(
    dataset_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    rank: int = 0,
    world_size: int = 1
):
    """
    Create a training-ready DataLoader with proper preprocessing.
    
    Args:
        dataset_path: Path to WebDataset directory
        batch_size: Batch size per GPU
        num_workers: Number of worker processes
        image_size: Target image size for training
        rank: Process rank for distributed training
        world_size: Total number of processes
        
    Returns:
        DataLoader ready for training
    """
    # Create image transforms
    image_transform = create_image_transform(image_size)
    
    # Create WebDataset loader
    loader = DistributedLLaVAWebDataset(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        buffer_size=1000,
        world_size=world_size,
        rank=rank
    )
    
    # Define preprocessing function
    def preprocess_sample(sample):
        """Preprocess a single sample for training."""
        try:
            key = sample["key"]
            image = sample["image"]
            json_data = sample["json"]
            
            # Apply image transforms
            if image_transform:
                image = image_transform(image)
            
            # Extract conversation data
            conversations = json_data.get("conversations", [])
            
            # Simple text processing (you'd want more sophisticated tokenization)
            human_text = ""
            assistant_text = ""
            
            for conv in conversations:
                role = conv.get("from", "")
                value = conv.get("value", "")
                
                if role == "human":
                    human_text = value.replace("<image>", "").strip()
                elif role == "gpt":
                    assistant_text = value.strip()
            
            return {
                "key": key,
                "image": image,
                "human_text": human_text,
                "assistant_text": assistant_text,
                "conversations": conversations
            }
            
        except Exception as e:
            if rank == 0:
                print(f"Error preprocessing sample: {e}")
            return None
    
    # Create dataset with preprocessing
    dataset = loader.create_dataset(decode_images=True, image_format="PIL")
    
    # Import webdataset for handler
    import webdataset as wds
    dataset = dataset.map(preprocess_sample, handler=wds.ignore_and_continue)
    dataset = dataset.select(lambda x: x is not None)
    
    # Custom collate function for training
    def training_collate_fn(samples):
        """Collate function optimized for training."""
        if not samples:
            return {}
        
        # Stack images into tensors
        images = torch.stack([s["image"] for s in samples])
        
        # Collect text data
        keys = [s["key"] for s in samples]
        human_texts = [s["human_text"] for s in samples]
        assistant_texts = [s["assistant_text"] for s in samples]
        conversations = [s["conversations"] for s in samples]
        
        return {
            "keys": keys,
            "images": images,  # [B, C, H, W] tensor
            "human_texts": human_texts,
            "assistant_texts": assistant_texts,
            "conversations": conversations
        }
    
    # Create batched dataset
    dataset = dataset.batched(batch_size, collation_fn=training_collate_fn)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=None,  # Batching handled by WebDataset
        num_workers=0,    # Use WebDataset's internal workers
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader


class DummyVisionLanguageModel(nn.Module):
    """Dummy model for demonstration purposes."""
    
    def __init__(self, vocab_size: int = 32000):
        super().__init__()
        # Vision encoder (dummy CNN)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512)
        )
        
        # Language model head (dummy)
        self.language_head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, vocab_size)
        )
    
    def forward(self, images, **kwargs):
        """Forward pass."""
        # Encode images
        vision_features = self.vision_encoder(images)
        
        # Generate language outputs (dummy)
        logits = self.language_head(vision_features)
        
        return logits


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rank: int = 0
):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        images = batch["images"].to(device)
        
        # Forward pass
        logits = model(images)
        
        # Dummy loss (replace with actual loss computation)
        target = torch.randint(0, logits.size(-1), (logits.size(0),), device=device)
        loss = nn.CrossEntropyLoss()(logits, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Log progress
        if rank == 0 and batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            print(f"  Image shape: {images.shape}")
            print(f"  Sample texts: {batch['human_texts'][0][:100]}...")
        
        # Demo - only train a few batches
        if batch_idx >= 20:
            break
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="LLaVA Training Example")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain/webdataset_output",
        help="Path to WebDataset directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Set device based on availability
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        if rank == 0:
            print(f"Using GPU training on {torch.cuda.device_count()} GPUs")
    else:
        device = torch.device('cpu')
        if rank == 0:
            print("Using CPU for training")
    
    if rank == 0:
        device_type = "GPUs" if torch.cuda.is_available() else "CPUs"
        print(f"🚀 Starting training on {world_size} {device_type}")
        print(f"📊 Dataset: {args.dataset_path}")
        print(f"🔧 Batch size: {args.batch_size} per process")
        print(f"🖼️  Image size: {args.image_size}x{args.image_size}")
        print(f"💻 Device: {device}")
    
    # Create dataloader
    print(f"📚 Creating dataloader (rank {rank})...")
    dataloader = create_training_dataloader(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        rank=rank,
        world_size=world_size
    )
    
    # Create model
    model = DummyVisionLanguageModel().to(device)
    
    # Wrap with DDP for distributed training
    if world_size > 1:
        if torch.cuda.is_available():
            # GPU training - use device_ids for proper GPU assignment
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            # CPU training without device_ids
            model = DDP(model)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    if rank == 0:
        print("✅ Model and optimizer created")
        print("🏋️  Starting training...")
    
    # Training loop (demo)
    for epoch in range(2):  # Just 2 epochs for demo
        if rank == 0:
            print(f"\\n📖 Epoch {epoch + 1}/2")
        
        avg_loss = train_epoch(model, dataloader, optimizer, device, rank)
        
        if rank == 0:
            print(f"📊 Epoch {epoch + 1} complete, Average Loss: {avg_loss:.4f}")
    
    if rank == 0:
        print("🎉 Training complete!")
    
    # Cleanup distributed
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()