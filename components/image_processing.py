# components/image_processing.py

"""
Enhanced Image Processing Module
Validates, processes, and extracts metadata from design images
Technology: PIL + CLIP + Base64 + NumPy + Hashlib
"""

import base64
import hashlib
from io import BytesIO
from PIL import Image
from typing import Optional, Dict, List, Tuple
import numpy as np
from dataclasses import dataclass, asdict
from collections import Counter

# Try to load CLIP - make it optional
clip = None
clip_model = None
clip_preprocess = None
device = "cpu"

try:
    import torch
    import clip as clip_module
    clip = clip_module
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        print(f"[OK] CLIP model loaded on {device}")
    except Exception as e:
        print(f"[WARN] Error loading CLIP model: {e}")
        clip_model, clip_preprocess = None, None
except ImportError:
    print("[WARN] CLIP module not installed. Image feature extraction disabled.")
    clip = None
    clip_model = None
    clip_preprocess = None

# Constants
ALLOWED_FORMATS = {'JPEG', 'PNG', 'GIF', 'BMP', 'WEBP', 'TIFF'}
MAX_FILE_SIZE_MB = 10  # 10MB limit
MAX_IMAGE_DIMENSION = 8000  # Max width or height
MIN_IMAGE_DIMENSION = 50   # Min width or height
THUMBNAIL_SIZE = (200, 200)


@dataclass
class ImageMetadata:
    """Structured image metadata"""
    width: int
    height: int
    aspect_ratio: float
    format: str
    color_mode: str
    file_size_bytes: int
    file_hash: str
    thumbnail_base64: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ProcessedImageResult:
    """Complete result of image processing"""
    original_image: Image.Image
    rgb_image: Image.Image
    thumbnail: Image.Image
    metadata: ImageMetadata
    embedding: Optional[np.ndarray] = None
    color_palette: Optional[List[Tuple[int, int, int]]] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def get_original_base64(self) -> str:
        """Get original image as base64"""
        return image_to_base64(self.rgb_image)

    def get_thumbnail_base64(self) -> str:
        """Get thumbnail as base64"""
        return image_to_base64(self.thumbnail)


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_image_file(file_bytes: bytes, filename: str, max_size_mb: int = MAX_FILE_SIZE_MB) -> Tuple[bool, Optional[str]]:
    """
    Validate image file before processing

    Args:
        file_bytes: Raw file bytes
        filename: Original filename
        max_size_mb: Maximum allowed file size in MB

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    # Check file size
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File size {file_size_mb:.1f}MB exceeds limit of {max_size_mb}MB"

    # Check file extension (case-insensitive)
    file_ext = filename.split('.')[-1].upper() if '.' in filename else ''
    # Map common extensions to standard formats
    ext_mapping = {
        'JPG': 'JPEG',
        'JPEG': 'JPEG',
        'PNG': 'PNG',
        'GIF': 'GIF',
        'BMP': 'BMP',
        'WEBP': 'WEBP',
        'TIFF': 'TIFF',
        'TIF': 'TIFF',
    }

    if file_ext not in ext_mapping:
        return False, f"File format .{file_ext} not supported. Allowed: {', '.join(sorted(ext_mapping.keys()))}"

    # Check if it's actually an image
    try:
        img = Image.open(BytesIO(file_bytes))
        img.verify()
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

    return True, None


def validate_image_dimensions(width: int, height: int) -> Tuple[bool, Optional[str]]:
    """
    Validate image dimensions

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        return False, f"Image too small: {width}x{height}. Minimum: {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}"

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        return False, f"Image too large: {width}x{height}. Maximum: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}"

    return True, None


# ============================================================================
# FILE PROCESSING FUNCTIONS
# ============================================================================

def compute_file_hash(file_bytes: bytes) -> str:
    """
    Compute SHA256 hash of file bytes

    Args:
        file_bytes: Raw file bytes

    Returns:
        str: SHA256 hash (hex string, first 16 chars)
    """
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def normalize_to_rgb(pil_image: Image.Image) -> Tuple[Image.Image, List[str]]:
    """
    Normalize image to RGB color mode, handling transparency

    Args:
        pil_image: PIL Image object

    Returns:
        Tuple[Image.Image, List[str]]: (rgb_image, warnings)
    """
    warnings = []

    if pil_image.mode == 'RGB':
        return pil_image, warnings

    if pil_image.mode in ('RGBA', 'LA', 'PA'):
        # Convert with alpha blending on white background
        warnings.append(
            f"Converted from {pil_image.mode} (removed transparency)")
        rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
        if pil_image.mode == 'PA':
            pil_image = pil_image.convert('RGBA')
        rgb_image.paste(pil_image, mask=pil_image.split()
                        [-1] if pil_image.mode in ('RGBA', 'LA') else None)
        return rgb_image, warnings

    if pil_image.mode == 'P':
        warnings.append("Converted from palette mode (P)")
        return pil_image.convert('RGB'), warnings

    if pil_image.mode == '1':
        warnings.append("Converted from binary mode (1)")
        return pil_image.convert('RGB'), warnings

    if pil_image.mode == 'L':
        warnings.append("Converted from grayscale mode (L)")
        return pil_image.convert('RGB'), warnings

    # Default: convert any other mode
    if pil_image.mode not in ('RGB',):
        warnings.append(f"Converted from {pil_image.mode}")
        return pil_image.convert('RGB'), warnings

    return pil_image, warnings


# ============================================================================
# IMAGE GENERATION FUNCTIONS
# ============================================================================

def generate_thumbnail(pil_image: Image.Image, size: Tuple[int, int] = THUMBNAIL_SIZE) -> Image.Image:
    """
    Generate thumbnail for UI display

    Args:
        pil_image: PIL Image object
        size: Thumbnail size (width, height)

    Returns:
        Image.Image: Thumbnail image
    """
    thumbnail = pil_image.copy()
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    return thumbnail


def extract_color_palette(pil_image: Image.Image, num_colors: int = 5) -> List[Tuple[int, int, int]]:
    """
    Extract top N most common colors from image

    Args:
        pil_image: PIL Image object (should be RGB)
        num_colors: Number of colors to extract

    Returns:
        List[Tuple[int, int, int]]: List of RGB color tuples, sorted by frequency
    """
    try:
        # Resize for faster processing
        temp_img = pil_image.copy()
        temp_img.thumbnail((100, 100))

        # Convert to RGB if needed
        if temp_img.mode != 'RGB':
            temp_img = temp_img.convert('RGB')

        # Get pixel data
        pixels = list(temp_img.getdata())

        # Count color frequency
        color_counter = Counter(pixels)

        # Get top N colors
        palette = [color for color, _ in color_counter.most_common(num_colors)]

        return palette
    except Exception as e:
        print(f"[WARN] Error extracting palette: {e}")
        return []


def image_to_base64(pil_image: Image.Image) -> str:
    """
    Convert PIL image to base64 string

    Args:
        pil_image: PIL.Image object

    Returns:
        str: Base64 encoded image string
    """
    try:
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=95)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_base64
    except Exception as e:
        raise Exception(f"Error converting image to base64: {str(e)}")


def generate_clip_embedding(pil_image: Image.Image) -> Optional[np.ndarray]:
    """
    Generate CLIP embedding for image similarity search

    Args:
        pil_image: PIL.Image object

    Returns:
        Optional[np.ndarray]: 512-dimensional CLIP embedding or None if CLIP unavailable
    """
    if clip_model is None or clip_preprocess is None:
        print("[WARN] CLIP not loaded, returning None")
        return None

    try:
        image_input = clip_preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            # Normalize for cosine similarity
            image_features = image_features / \
                image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()[0]
    except Exception as e:
        print(f"[WARN] Error generating CLIP embedding: {e}")
        # Fallback to simple hash-based embedding
        return generate_fallback_embedding(pil_image)


def generate_fallback_embedding(pil_image: Image.Image) -> np.ndarray:
    """
    Generate a simple fallback embedding when CLIP is unavailable.
    Uses image histogram features for similarity search.
    
    Args:
        pil_image: PIL.Image object
        
    Returns:
        np.ndarray: 512-dimensional feature vector
    """
    try:
        # Resize to consistent size
        img_resized = pil_image.resize((64, 64))
        img_array = np.array(img_resized).flatten().astype(np.float32)
        
        # Pad or truncate to 512 dimensions
        if len(img_array) >= 512:
            embedding = img_array[:512]
        else:
            embedding = np.pad(img_array, (0, 512 - len(img_array)), mode='constant')
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding
    except Exception as e:
        print(f"[WARN] Fallback embedding failed: {e}")
        # Return zero vector as last resort
        return np.zeros(512, dtype=np.float32)


# ============================================================================
# METADATA EXTRACTION
# ============================================================================

def extract_metadata(pil_image: Image.Image, file_bytes: bytes, filename: str) -> ImageMetadata:
    """
    Extract comprehensive image metadata

    Args:
        pil_image: PIL Image object (should be RGB)
        file_bytes: Original file bytes
        filename: Original filename

    Returns:
        ImageMetadata: Structured metadata
    """
    file_hash = compute_file_hash(file_bytes)
    thumbnail = generate_thumbnail(pil_image)
    thumbnail_base64 = image_to_base64(thumbnail)

    aspect_ratio = round(pil_image.width / pil_image.height,
                         2) if pil_image.height > 0 else 0

    return ImageMetadata(
        width=pil_image.width,
        height=pil_image.height,
        aspect_ratio=aspect_ratio,
        format=pil_image.format if pil_image.format else "JPEG",
        color_mode=pil_image.mode,
        file_size_bytes=len(file_bytes),
        file_hash=file_hash,
        thumbnail_base64=thumbnail_base64
    )


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_image_file(file_bytes: bytes, filename: str, extract_palette: bool = True) -> Tuple[ProcessedImageResult, Optional[str]]:
    """
    Complete image processing pipeline with validation

    Args:
        file_bytes: Raw image file bytes
        filename: Original filename
        extract_palette: Whether to extract color palette (default: True)

    Returns:
        Tuple[ProcessedImageResult, Optional[str]]: (result, error_message)
        Returns (None, error_message) if processing fails
    """
    # Step 1: Validate file
    is_valid, error = validate_image_file(file_bytes, filename)
    if not is_valid:
        return None, error

    # Step 2: Open image
    try:
        original_image = Image.open(BytesIO(file_bytes))
    except Exception as e:
        return None, f"Failed to open image: {str(e)}"

    # Step 3: Validate dimensions
    is_valid, error = validate_image_dimensions(
        original_image.width, original_image.height)
    if not is_valid:
        return None, error

    # Step 4: Normalize to RGB
    rgb_image, norm_warnings = normalize_to_rgb(original_image)

    # Step 5: Extract metadata
    metadata = extract_metadata(rgb_image, file_bytes, filename)

    # Step 6: Generate thumbnail
    thumbnail = generate_thumbnail(rgb_image)

    # Step 7: Generate CLIP embedding
    embedding = generate_clip_embedding(rgb_image)

    # Step 8: Extract palette
    palette = extract_color_palette(rgb_image) if extract_palette else None

    result = ProcessedImageResult(
        original_image=original_image,
        rgb_image=rgb_image,
        thumbnail=thumbnail,
        metadata=metadata,
        embedding=embedding,
        color_palette=palette,
        warnings=norm_warnings
    )

    return result, None


# ============================================================================
# LEGACY FUNCTIONS (FOR BACKWARD COMPATIBILITY)
# ============================================================================

def preprocess_image(uploaded_file):
    """
    Legacy function: Convert uploaded file to PIL Image

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        PIL.Image: Processed image (resized to max 1024x1024)
    """
    try:
        image = Image.open(uploaded_file)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize for API efficiency (max 1024x1024)
        max_size = 1024
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        return image
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")


def extract_image_metadata(pil_image: Image.Image) -> Dict:
    """
    Legacy function: Extract technical image properties

    Args:
        pil_image: PIL.Image object

    Returns:
        dict: Image metadata (dimensions, format, aspect ratio, etc.)
    """
    try:
        return {
            "width": pil_image.width,
            "height": pil_image.height,
            "format": pil_image.format if pil_image.format else "JPEG",
            "mode": pil_image.mode,
            "aspect_ratio": round(pil_image.width / pil_image.height, 2)
        }
    except Exception as e:
        return {"error": str(e)}
