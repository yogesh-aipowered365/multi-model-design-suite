"""
Visual Feedback Generation System
Generate annotated images, mockups, and before/after comparisons
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import io
import base64
import numpy as np
# import cv2  # Commented out due to NumPy compatibility issues


def create_annotated_design(image_base64, recommendations, annotation_type="overlay"):
    """
    Create annotated version of design with issues highlighted

    Args:
        image_base64: Original image
        recommendations: List of recommendations with location data
        annotation_type: "overlay", "arrows", "heatmap"

    Returns:
        PIL.Image: Annotated image
    """
    # Decode image
    img_data = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(img_data)).convert('RGB')

    # Create drawing layer
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Try to load font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Define annotation colors
    priority_colors = {
        'critical': (255, 0, 0, 180),      # Red
        'high': (255, 165, 0, 180),        # Orange
        'medium': (255, 255, 0, 180),      # Yellow
        'low': (0, 255, 0, 180)            # Green
    }

    # Annotate based on recommendations
    for i, rec in enumerate(recommendations[:10], 1):  # Top 10
        priority = rec.get('priority', 'medium')
        color = priority_colors.get(priority, (255, 255, 0, 180))

        # Simulate location (in real implementation, this would come from rec)
        # For demo, place annotations around the image
        x = (i % 3) * (img.width // 3) + 50
        y = ((i // 3) * (img.height // 4)) + 50

        if annotation_type == "overlay":
            # Draw circle marker
            marker_radius = 25
            draw.ellipse([x-marker_radius, y-marker_radius,
                         x+marker_radius, y+marker_radius],
                         fill=color, outline=(255, 255, 255, 255), width=3)

            # Draw number
            number_text = str(i)
            bbox = draw.textbbox((0, 0), number_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text((x - text_width//2, y - text_height//2),
                      number_text, fill=(255, 255, 255, 255), font=font)

            # Draw line to annotation box
            box_x = x + 100
            box_y = y
            draw.line([x + marker_radius, y, box_x, box_y],
                      fill=color, width=2)

            # Draw annotation text box
            issue_title = rec.get('issue', {}).get('title', 'Issue')[:30]
            box_width = 200
            box_height = 50

            draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height],
                           fill=color, outline=(255, 255, 255, 255), width=2)

            draw.text((box_x + 10, box_y + 5), f"#{i}: {issue_title}",
                      fill=(255, 255, 255, 255), font=font_small)
            draw.text((box_x + 10, box_y + 25), f"Priority: {priority.upper()}",
                      fill=(255, 255, 255, 255), font=font_small)

    # Composite overlay onto original
    img_rgba = img.convert('RGBA')
    annotated = Image.alpha_composite(img_rgba, overlay)

    return annotated.convert('RGB')


def generate_before_after_mockup(image_base64, recommendations):
    """
    Create before/after comparison mockup

    Args:
        image_base64: Original image
        recommendations: List of recommendations to apply

    Returns:
        PIL.Image: Side-by-side before/after image
    """
    # Decode original
    img_data = base64.b64decode(image_base64)
    original = Image.open(io.BytesIO(img_data)).convert('RGB')

    # Create "improved" version (simulated improvements)
    improved = original.copy()

    # Apply visual improvements (simulated)
    # 1. Increase contrast slightly
    enhancer = ImageEnhance.Contrast(improved)
    improved = enhancer.enhance(1.2)

    # 2. Increase sharpness
    enhancer = ImageEnhance.Sharpness(improved)
    improved = enhancer.enhance(1.3)

    # 3. Adjust colors based on recommendations
    for rec in recommendations:
        category = rec.get('category', '')
        if 'color' in category.lower():
            # Slightly adjust saturation
            enhancer = ImageEnhance.Color(improved)
            improved = enhancer.enhance(1.1)
            break

    # Create side-by-side comparison
    width, height = original.size
    combined = Image.new('RGB', (width * 2 + 40, height + 100), 'white')
    draw = ImageDraw.Draw(combined)

    # Load font
    try:
        font_large = ImageFont.truetype("arial.ttf", 32)
        font_small = ImageFont.truetype("arial.ttf", 18)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Paste images
    combined.paste(original, (0, 80))
    combined.paste(improved, (width + 40, 80))

    # Add labels
    draw.text((width//2 - 50, 20), "BEFORE", fill='red', font=font_large)
    draw.text((width + 40 + width//2 - 50, 20),
              "AFTER", fill='green', font=font_large)

    # Add divider line
    draw.line([(width + 20, 0), (width + 20, height + 100)],
              fill='gray', width=3)

    # Add improvement summary at bottom
    num_improvements = len(recommendations)
    summary = f"Applied {num_improvements} recommendations"
    draw.text((combined.width//2 - 100, height + 85),
              summary, fill='black', font=font_small)

    return combined


def generate_heatmap_visualization(image_base64, analysis_type="attention"):
    """
    Generate attention/problem heatmap overlay
    Args:
        image_base64: Original image
        analysis_type: "attention" (where users look) or "problems" (issue density)

    Returns:
        PIL.Image: Image with heatmap overlay
    """
    # Decode image
    img_data = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    img_np = np.array(img)

    # Create heatmap (simulated)
    height, width = img_np.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)

    if analysis_type == "attention":
        # Simulate F-pattern attention (top-left bias)
        y_coords, x_coords = np.ogrid[:height, :width]

        # Top-left has highest attention
        heatmap = np.exp(-((x_coords / width) ** 2 +
                         (y_coords / height) ** 2) * 2)

        # Add horizontal band (F-pattern)
        heatmap[height//4:height//3, :] += 0.5

    else:  # problems
        # Simulate problem areas (random for demo)
        num_problems = 5
        for _ in range(num_problems):
            cx = np.random.randint(width//4, 3*width//4)
            cy = np.random.randint(height//4, 3*height//4)
            radius = min(width, height) // 6

            y_coords, x_coords = np.ogrid[:height, :width]
            mask = ((x_coords - cx)**2 + (y_coords - cy)**2) <= radius**2
            heatmap[mask] += 0.3

    # Normalize
    heatmap = (heatmap - heatmap.min()) / \
        (heatmap.max() - heatmap.min() + 1e-8)

    # Convert to color heatmap using PIL colormap
    heatmap_255 = (heatmap * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_255, mode='L')
    heatmap_img = heatmap_img.convert('RGB')

    # Apply a simple color mapping (blue to red gradient)
    heatmap_data = np.array(heatmap_img)
    heatmap_colored = np.zeros((*heatmap_255.shape, 3), dtype=np.uint8)
    heatmap_colored[:, :, 0] = heatmap_255  # Red channel
    heatmap_colored[:, :, 1] = 128 - heatmap_255 // 2  # Green channel
    heatmap_colored[:, :, 2] = 255 - heatmap_255  # Blue channel

    # Blend with original using PIL
    alpha = 0.5
    img_result = Image.fromarray(img_np)
    heatmap_img_rgb = Image.fromarray(heatmap_colored)

    # Blend using PIL's alpha_composite
    img_with_alpha = img_result.convert('RGBA')
    heatmap_with_alpha = heatmap_img_rgb.convert('RGBA')
    heatmap_with_alpha.putalpha(int(256 * alpha))

    blended = Image.alpha_composite(
        img_with_alpha, heatmap_with_alpha).convert('RGB')
    blended = np.array(blended)

    result = Image.fromarray(blended)

    # Add legend
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    legend_text = "High" if analysis_type == "attention" else "Problem Areas"
    draw.rectangle([10, height - 40, 150, height - 10], fill='red')
    draw.text((15, height - 35), legend_text, fill='white', font=font)

    legend_text2 = "Low" if analysis_type == "attention" else "No Issues"
    draw.rectangle([10, height - 70, 150, height - 45], fill='blue')
    draw.text((15, height - 65), legend_text2, fill='white', font=font)

    return result


def generate_color_palette_visualization(colors_hex_list):
    """
    Create visual color palette from extracted colors
    Args:
    colors_hex_list: List of hex color codes

    Returns:
        PIL.Image: Color palette visualization
    """
    num_colors = len(colors_hex_list)

    if num_colors == 0:
        return None

    # Create image
    swatch_width = 100
    swatch_height = 100
    img_width = swatch_width * num_colors
    img_height = swatch_height + 50

    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    # Draw color swatches
    for i, hex_color in enumerate(colors_hex_list):
        x = i * swatch_width

        # Convert hex to RGB
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[j:j+2], 16) for j in (0, 2, 4))

        # Draw swatch
        draw.rectangle([x, 0, x + swatch_width, swatch_height], fill=rgb)

        # Draw hex code below
        text = f"#{hex_color.upper()}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (swatch_width - text_width) // 2

        draw.text((text_x, swatch_height + 15), text, fill='black', font=font)

    return img


def image_to_base64(pil_image):
    """
    Convert PIL image to base64 string
    Args:
        pil_image: PIL.Image object

    Returns:
        str: Base64 encoded string
    """
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str
