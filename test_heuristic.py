import os
from PIL import Image
import numpy as np

def validate_chest_xray(image_path):
    try:
        img = Image.open(image_path)
        
        # 1. Format/Dimensions check
        width, height = img.size
        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False, f"Invalid aspect ratio: {aspect_ratio:.2f}. Chest X-rays should be roughly square."
        
        # Convert to RGB to analyze colors
        img_rgb = img.convert('RGB')
        img_np = np.array(img_rgb)
        
        # 2. Color check: standard deviation across color channels
        # For grayscale images, R, G, B channels are identical or very close.
        # Compute mean absolute difference between channels.
        rg_diff = np.abs(img_np[:, :, 0].astype(float) - img_np[:, :, 1].astype(float))
        gb_diff = np.abs(img_np[:, :, 1].astype(float) - img_np[:, :, 2].astype(float))
        br_diff = np.abs(img_np[:, :, 2].astype(float) - img_np[:, :, 0].astype(float))
        color_diff = (rg_diff.mean() + gb_diff.mean() + br_diff.mean()) / 3.0
        
        if color_diff > 12.0:
            return False, f"Image is colored (color variance: {color_diff:.2f}). Chest X-rays must be grayscale."
            
        # 3. Contrast check: X-rays have high contrast (ribs/spine/lungs)
        # Check standard deviation of brightness
        gray_img = img.convert('L')
        gray_np = np.array(gray_img)
        std_dev = gray_np.std()
        
        if std_dev < 15.0:
            return False, f"Image contrast is too low (std dev: {std_dev:.2f}). This does not appear to be an X-ray."
            
        # 4. Check brightness distribution (borders vs center)
        # Chest X-rays have a dark background margin and a brighter center (lungs/ribs/spine).
        h, w = gray_np.shape
        border_w = int(w * 0.1)
        border_h = int(h * 0.1)
        
        # Compute average of borders (top, bottom, left, right 10%)
        border_pixels = []
        border_pixels.extend(gray_np[:border_h, :].flatten()) # top
        border_pixels.extend(gray_np[-border_h:, :].flatten()) # bottom
        border_pixels.extend(gray_np[:, :border_w].flatten()) # left
        border_pixels.extend(gray_np[:, -border_w:].flatten()) # right
        mean_border = np.mean(border_pixels)
        
        # Compute center (middle 50%)
        center_y_start = int(h * 0.25)
        center_y_end = int(h * 0.75)
        center_x_start = int(w * 0.25)
        center_x_end = int(w * 0.75)
        center_pixels = gray_np[center_y_start:center_y_end, center_x_start:center_x_end]
        mean_center = np.mean(center_pixels)
        
        # Chest X-rays are brighter in the center than on the borders
        # Check if center is brighter, or at least not completely uniform
        if mean_center < 35.0:
             return False, "Image is too dark (average center brightness is extremely low)."
             
        # Also check if it's completely white/bright
        if mean_center > 240.0:
             return False, "Image is too bright (overexposed or plain white)."
             
        return True, "Valid chest X-ray."
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

# Test on a few dummy images
if __name__ == "__main__":
    dummy_dir = "data/Lung Disease Dataset/train/Normal"
    count = 0
    for img_name in os.listdir(dummy_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(dummy_dir, img_name)
            res, msg = validate_chest_xray(path)
            print(f"{img_name}: {res} ({msg})")
            count += 1
            if count >= 10:
                break
