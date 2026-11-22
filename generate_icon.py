"""Generate high-resolution application icons for agriGPT."""
import os
from PIL import Image, ImageDraw, ImageFont

def create_agrigpt_icon(output_dir="assets"):
    os.makedirs(output_dir, exist_ok=True)
    size = (512, 512)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 1. Outer circular badge with emerald green gradient effect
    draw.ellipse([(16, 16), (496, 496)], fill=(46, 125, 50, 255), outline=(27, 94, 32, 255), width=12)
    # Inner subtle rim
    draw.ellipse([(32, 32), (480, 480)], outline=(129, 199, 132, 220), width=6)

    # 2. Draw stylized plant stalk / leaf in golden wheat & light green
    # Main stem
    stem_points = [(256, 420), (256, 220), (256, 130)]
    draw.line(stem_points, fill=(245, 245, 245, 255), width=16)

    # Left leaf
    left_leaf = [(256, 310), (160, 260), (140, 180), (210, 190), (256, 260)]
    draw.polygon(left_leaf, fill=(165, 214, 167, 255))
    draw.line([(256, 310), (140, 180)], fill=(46, 125, 50, 255), width=4)

    # Right leaf
    right_leaf = [(256, 270), (352, 220), (372, 140), (302, 150), (256, 220)]
    draw.polygon(right_leaf, fill=(255, 213, 79, 255)) # Golden wheat tone
    draw.line([(256, 270), (372, 140)], fill=(245, 124, 0, 255), width=4)

    # Center top leaf bud
    top_bud = [(256, 140), (230, 80), (256, 50), (282, 80), (256, 140)]
    draw.polygon(top_bud, fill=(200, 230, 201, 255))

    # Base soil mound
    draw.chord([(120, 370), (392, 450)], start=0, end=180, fill=(27, 94, 32, 255))

    # Save PNG
    png_path = os.path.join(output_dir, "icon.png")
    img.save(png_path, format="PNG")

    # Save multi-resolution Windows ICO
    ico_path = os.path.join(output_dir, "agriGPT.ico")
    icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(ico_path, format="ICO", sizes=icon_sizes)

    print(f"Generated {png_path} and {ico_path}")

if __name__ == "__main__":
    create_agrigpt_icon()
