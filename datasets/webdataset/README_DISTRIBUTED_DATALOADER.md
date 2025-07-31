# Distributed LLaVA WebDataset Loader

### 1. Basic Data Loading Test

```bash
# Test Loading data
python distributed_webdataset_loader.py --mode test --batch_size 16
```

### 2. Dataset Resolution Analysis

```bash
# Analyze resolution distribution
python distributed_webdataset_loader.py --mode analyze --max_samples 50000
```

### 3. Training Pipeline Demo

```bash
# Example of training with the dataloader
python training_example.py --batch_size 8 --image_size 224
```


```bash
# Distributed training test
torchrun --nproc_per_node=2 training_example.py --num_workers 8 --batch_size 128 --image_size 224
```
