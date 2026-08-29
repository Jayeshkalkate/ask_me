# create_icons.py - Run this to generate PWA icons
import os
from PIL import Image, ImageDraw

def create_icon(size, filename):
    """Create a simple icon with ASK_ME text"""
    img = Image.new('RGB', (size, size), color='#4f46e5')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple design
    draw.rectangle([10, 10, size-10, size-10], outline='#818cf8', width=4)
    
    # Add text
    text = "AM"
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", size//3)
    except:
        font = ImageFont.load_default()
    
    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size - text_width) // 2
    y = (size - text_height) // 2
    
    draw.text((x, y), text, fill='white', font=font)
    
    # Save
    os.makedirs('static/img/icons', exist_ok=True)
    img.save(f'static/img/icons/{filename}')
    print(f"Created {filename}")

# Create all sizes
sizes = [
    (72, 'icon-72x72.png'),
    (96, 'icon-96x96.png'),
    (128, 'icon-128x128.png'),
    (144, 'icon-144x144.png'),
    (152, 'icon-152x152.png'),
    (192, 'icon-192x192.png'),
    (384, 'icon-384x384.png'),
    (512, 'icon-512x512.png'),
]

print("Generating PWA icons...")
for size, filename in sizes:
    create_icon(size, filename)
print("✅ All icons created!")