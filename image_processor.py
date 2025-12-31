"""Image processing for RAG system (no OCR)."""

from pathlib import Path
from typing import Dict, Optional, List
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class ImageProcessor:
    """Processes images for embedding."""

    def __init__(self, max_size: int = 1024):
        self.max_size = max_size
        if not PIL_AVAILABLE:
            raise ImportError("PIL required: pip install Pillow")

    def load_image(self, filepath: str) -> Image.Image:
        """Load and preprocess image."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {filepath}")

        img = Image.open(filepath)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        if max(img.size) > self.max_size:
            img.thumbnail((self.max_size, self.max_size), Image.Resampling.LANCZOS)
        return img

    def detect_image_type(self, img: Image.Image) -> str:
        """Detect image type: diagram, screenshot, or photo."""
        img_array = np.array(img.convert('RGB'))
        colors = img_array.reshape(-1, 3)
        color_ratio = len(np.unique(colors, axis=0)) / colors.shape[0]

        gray = np.mean(img_array, axis=2)
        edge_density = (np.mean(np.abs(np.diff(gray, axis=1))) +
                       np.mean(np.abs(np.diff(gray, axis=0)))) / 2

        if color_ratio < 0.01:
            return "diagram"
        elif edge_density > 30:
            return "screenshot"
        elif color_ratio > 0.3:
            return "photo"
        return "diagram"

    def generate_image_description(self, img: Image.Image) -> str:
        """Generate basic image description."""
        width, height = img.size
        ratio = width / height

        if ratio > 1.5:
            layout = "wide horizontal"
        elif ratio < 0.67:
            layout = "tall vertical"
        else:
            layout = "square"

        img_type = self.detect_image_type(img)
        return f"{layout} {img_type}, {width}x{height}px"

    def process_image(self, filepath: str, metadata: Optional[Dict] = None) -> Dict:
        """Process image for RAG system."""
        img = self.load_image(filepath)
        return {
            'type': 'image',
            'filepath': str(filepath),
            'content': '',  # No OCR text
            'description': self.generate_image_description(img),
            'metadata': metadata or {},
            'dimensions': img.size,
        }

    def process_directory(self, dirpath: str, patterns: List[str] = None,
                         metadata: Optional[Dict] = None) -> List[Dict]:
        """Process all images in a directory."""
        if patterns is None:
            patterns = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp"]

        path = Path(dirpath)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dirpath}")

        results = []
        for pattern in patterns:
            for filepath in path.glob(pattern):
                try:
                    results.append(self.process_image(str(filepath), metadata))
                except Exception as e:
                    print(f"Failed to process {filepath}: {e}")
        return results
