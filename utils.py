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
                images.append(Image.open(file_path))
                image_names.append(file_path.stem)  # filename without extension
                image_paths.append(str(file_path))
                print(f"Loaded: {file_path.name}")
            except Exception as e:
                print(f"Failed to load {file_path.name}: {e}")
    
    return images, image_names, image_paths
