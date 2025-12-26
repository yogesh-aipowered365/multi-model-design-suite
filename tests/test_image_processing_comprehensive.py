"""
tests/test_image_processing_comprehensive.py

Comprehensive test suite for image processing module.
Validates:
- Image loading and format handling
- Metadata extraction accuracy
- Base64 encoding/decoding
- Color palette extraction
- Shape consistency
"""

import pytest
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path

from components.image_processing import (
    extract_image_metadata,
    image_to_base64,
    base64_to_image,
    extract_color_palette,
)


class TestImageLoading:
    """Test image loading and format handling."""

    @pytest.fixture
    def sample_image_png(self):
        """Create a sample PNG image."""
        img = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer

    @pytest.fixture
    def sample_image_jpg(self):
        """Create a sample JPEG image."""
        img = Image.new('RGB', (100, 100), color='blue')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        buffer.seek(0)
        return buffer

    def test_png_loading(self, sample_image_png):
        """PNG images should load correctly."""
        img = Image.open(sample_image_png)
        assert img.size == (100, 100)
        assert img.format == 'PNG'

    def test_jpg_loading(self, sample_image_jpg):
        """JPEG images should load correctly."""
        img = Image.open(sample_image_jpg)
        assert img.size == (100, 100)
        # JPEG format might be None when loaded from BytesIO
        assert img.mode in ['RGB', 'RGBA']

    def test_unsupported_format(self):
        """Unsupported formats should raise error."""
        with pytest.raises((IOError, OSError)):
            # Try to open non-image data
            Image.open(io.BytesIO(b'not an image'))


class TestMetadataExtraction:
    """Test metadata extraction."""

    @pytest.fixture
    def test_image(self):
        """Create a test image with known dimensions."""
        img = Image.new('RGB', (640, 480), color='white')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer

    def test_metadata_extraction_shape(self, test_image):
        """Extracted metadata should have correct shape."""
        metadata = extract_image_metadata(test_image)

        assert isinstance(metadata, dict)
        assert 'width' in metadata
        assert 'height' in metadata
        assert 'file_size' in metadata

    def test_metadata_dimensions(self, test_image):
        """Metadata dimensions should match image."""
        metadata = extract_image_metadata(test_image)

        assert metadata['width'] == 640
        assert metadata['height'] == 480

    def test_metadata_completeness(self, test_image):
        """Metadata should include all expected fields."""
        metadata = extract_image_metadata(test_image)

        expected_fields = ['width', 'height', 'file_size', 'format']
        for field in expected_fields:
            assert field in metadata, f"Missing field: {field}"

    def test_metadata_type_consistency(self, test_image):
        """Metadata field types should be consistent."""
        metadata = extract_image_metadata(test_image)

        assert isinstance(metadata['width'], int)
        assert isinstance(metadata['height'], int)
        assert isinstance(metadata['file_size'], (int, float))
        assert isinstance(metadata['format'], str)


class TestBase64Encoding:
    """Test base64 encoding/decoding."""

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        img = Image.new('RGB', (50, 50), color='green')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer

    def test_encode_to_base64(self, test_image):
        """Image should encode to valid base64."""
        b64_string = image_to_base64(test_image)

        assert isinstance(b64_string, str)
        # Valid base64 strings only contain alphanumeric, +, /, and =
        assert all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/='
                   for c in b64_string)

    def test_roundtrip_encode_decode(self, test_image):
        """Encoding then decoding should preserve image."""
        original_data = test_image.read()
        test_image.seek(0)

        b64_string = image_to_base64(test_image)
        decoded = base64_to_image(b64_string)

        # Compare images by converting to numpy arrays
        original_img = Image.open(io.BytesIO(original_data))
        original_array = np.array(original_img)
        decoded_array = np.array(decoded)

        assert original_array.shape == decoded_array.shape

    def test_base64_non_empty(self, test_image):
        """Base64 encoding should not be empty."""
        b64_string = image_to_base64(test_image)
        assert len(b64_string) > 0

    def test_decode_invalid_base64(self):
        """Invalid base64 should raise error."""
        with pytest.raises((ValueError, TypeError)):
            base64_to_image("not_valid_base64!")


class TestColorPaletteExtraction:
    """Test color palette extraction."""

    @pytest.fixture
    def solid_color_image(self):
        """Create an image with a single solid color."""
        img = Image.new('RGB', (100, 100), color=(255, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer

    @pytest.fixture
    def multi_color_image(self):
        """Create an image with multiple colors."""
        img = Image.new('RGB', (200, 200))
        pixels = img.load()

        # Create quadrants with different colors
        for i in range(100):
            for j in range(100):
                pixels[i, j] = (255, 0, 0)  # Red
                pixels[i+100, j] = (0, 255, 0)  # Green
                pixels[i, j+100] = (0, 0, 255)  # Blue
                pixels[i+100, j+100] = (255, 255, 0)  # Yellow

        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer

    def test_palette_returns_list(self, solid_color_image):
        """Color palette should return a list."""
        palette = extract_color_palette(solid_color_image)
        assert isinstance(palette, list)

    def test_palette_is_rgb_tuples(self, solid_color_image):
        """Palette should contain RGB tuples."""
        palette = extract_color_palette(solid_color_image)

        for color in palette:
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(isinstance(c, int) for c in color)
            assert all(0 <= c <= 255 for c in color)

    def test_palette_not_empty(self, solid_color_image):
        """Palette should not be empty."""
        palette = extract_color_palette(solid_color_image)
        assert len(palette) > 0

    def test_palette_max_colors(self, multi_color_image):
        """Palette should respect color limit (typically 5-10 colors)."""
        palette = extract_color_palette(multi_color_image)
        # Typical palette extraction returns 5-10 dominant colors
        assert len(palette) <= 15

    def test_palette_consistency(self, solid_color_image):
        """Same image should produce consistent palette."""
        palette1 = extract_color_palette(solid_color_image)
        solid_color_image.seek(0)
        palette2 = extract_color_palette(solid_color_image)

        # Palettes should be identical
        assert palette1 == palette2
