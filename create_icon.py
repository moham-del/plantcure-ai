from PIL import Image, ImageDraw
import os

def create_icon(size):
    # Green background
    img = Image.new('RGB', (size, size), color='#0f4c1e')
    draw = ImageDraw.Draw(img)
    
    # Circle
    margin = size // 10
    draw.ellipse(
        [margin, margin, size-margin, size-margin],
        fill='#2ecc71'
    )
    
    # Inner circle
    margin2 = size // 4
    draw.ellipse(
        [margin2, margin2, size-margin2, size-margin2],
        fill='#1a7a3c'
    )
    
    # Save
    os.makedirs('static/images', exist_ok=True)
    img.save(f'static/images/icon-{size}.png')
    print(f'✅ Icon {size}x{size} created!')

create_icon(192)
create_icon(512)
print('🎉 All icons ready!')

