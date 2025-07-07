from pathlib import Path
from PIL import Image

# Simple function to load all images from a folder
def load_all_images(folder_path):
    """Load all image files from the specified folder."""
    folder = Path(folder_path)
    image_extensions = {'.png', '.jpg', '.jpeg'}
    
    images = []
    image_names = []
    image_paths = []
    
    # Get all files and filter for images
    for file_path in sorted(folder.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            try:
                images.append(Image.open(file_path).convert("RGB") )
                image_names.append(file_path.stem)  # filename without extension
                image_paths.append(str(file_path))
                print(f"Loaded: {file_path.name}")
            except Exception as e:
                print(f"Failed to load {file_path.name}: {e}")
    
    return images, image_names, image_paths


def split_into_quadrants(image_path):
    original_img = Image.open(image_path).convert("RGB")
    W, H = original_img.size
    print(f"Original size: {W}×{H}")
    
    square_size = min(W, H)
    left = (W - square_size) // 2
    top = (H - square_size) // 2
    square_img = original_img.crop((left, top, left + square_size, top + square_size))

    W, H = square_img.size
    W_half, H_half = W // 2, H // 2
    print(f"Each quadrant: {W_half}×{H_half}")
    
    quadrants = [
        square_img.crop((0, 0, W_half, H_half)),           # Top-left
        square_img.crop((W_half, 0, W, H_half)),           # Top-right
        square_img.crop((0, H_half, W_half, H)),           # Bottom-left
        square_img.crop((W_half, H_half, W, H)),           # Bottom-right
    ]

    return square_img, quadrants