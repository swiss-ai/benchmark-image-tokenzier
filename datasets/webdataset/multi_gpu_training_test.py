#!/usr/bin/env python3

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torchvision import transforms
from generalized_webdataset_loader import GeneralizedWebDatasetLoader, DatasetType


class SimpleMultiModalModel(nn.Module):
    """Simple multi-modal model for testing."""
    
    def __init__(self, image_size=224, hidden_dim=512, num_classes=10):
        super().__init__()
        # Simple CNN for image encoding
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, images):
        # Encode images
        features = self.image_encoder(images)
        features = features.flatten(1)
        output = self.projection(features)
        return output


def setup_distributed(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    rank,
    epoch,
    dataset_type
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0
    batch_count = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # Handle different dataset types
        if dataset_type == DatasetType.LLAVA:
            images = batch['images']
            # For LLaVA, we'll use dummy labels for testing
            labels = torch.randint(0, 10, (len(images),)).to(device)
        elif dataset_type == DatasetType.COCO:
            images = batch['images']
            # For COCO, we'll use dummy labels for testing
            labels = torch.randint(0, 10, (len(images),)).to(device)
        else:
            images = batch['images']
            labels = torch.randint(0, 10, (len(images),)).to(device)
        
        # Convert PIL images to tensors if needed
        if hasattr(images[0], 'size'):  # PIL Image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            images = torch.stack([transform(img) for img in images])
        
        images = images.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        batch_count += 1
        
        # Print progress
        if batch_idx % 10 == 0 and rank == 0:
            elapsed = time.time() - start_time
            samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
            print(f"[Epoch {epoch}][Batch {batch_idx}] "
                  f"Loss: {loss.item():.4f} | "
                  f"Speed: {samples_per_sec:.1f} samples/s")
        
        # Limit batches for testing
        if batch_idx >= 20:
            break
    
    # Gather metrics from all processes
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    
    if dist.is_initialized():
        avg_loss_tensor = torch.tensor(avg_loss).to(device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()
    
    return avg_loss, total_samples


def run_training(
    rank,
    world_size,
    dataset_path,
    dataset_type,
    batch_size,
    num_workers,
    num_epochs
):
    """Main training function for each process."""
    
    # Setup distributed
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    print(f"[Rank {rank}] Starting training on GPU {rank}")
    
    # Create dataloader
    loader = GeneralizedWebDatasetLoader(
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # Create dataset with transforms
    dataset = loader.create_dataset(
        decode_images=True,
        image_format="PIL",
        return_metadata=False
    )
    
    # Create dataloader
    dataloader = loader.create_dataloader(dataset)
    
    # Create model
    model = SimpleMultiModalModel().to(device)
    model = DDP(model, device_ids=[rank])
    
    # Create optimizer and loss
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")
        
        avg_loss, samples = train_epoch(
            model,
            dataloader,
            optimizer,
            criterion,
            device,
            rank,
            epoch + 1,
            loader.dataset_type
        )
        
        if rank == 0:
            print(f"\n[Epoch {epoch + 1}] Average Loss: {avg_loss:.4f}")
            print(f"[Epoch {epoch + 1}] Samples Processed: {samples * world_size}")
    
    # Cleanup
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Training Test")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to WebDataset directory"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["llava", "coco", "generic"],
        default="generic",
        help="Type of dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers per GPU"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)"
    )
    
    args = parser.parse_args()
    
    # Determine world size
    if args.world_size is None:
        world_size = torch.cuda.device_count()
    else:
        world_size = min(args.world_size, torch.cuda.device_count())
    
    if world_size == 0:
        print("No GPUs available. Please run on a machine with GPUs.")
        return
    
    print(f"\n{'='*60}")
    print(f"Multi-GPU Training Test")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Dataset Type: {args.dataset_type}")
    print(f"World Size: {world_size} GPUs")
    print(f"Batch Size per GPU: {args.batch_size}")
    print(f"Total Batch Size: {args.batch_size * world_size}")
    print(f"Workers per GPU: {args.num_workers}")
    print(f"Epochs: {args.num_epochs}")
    print(f"{'='*60}\n")
    
    # Launch training processes
    if world_size > 1:
        mp.spawn(
            run_training,
            args=(
                world_size,
                args.dataset_path,
                args.dataset_type,
                args.batch_size,
                args.num_workers,
                args.num_epochs
            ),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU training
        run_training(
            0,
            1,
            args.dataset_path,
            args.dataset_type,
            args.batch_size,
            args.num_workers,
            args.num_epochs
        )
    
    print("\n✅ Multi-GPU training test completed!")


if __name__ == "__main__":
    main()