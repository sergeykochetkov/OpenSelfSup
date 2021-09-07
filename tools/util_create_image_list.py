import os
import cv2
from PIL import Image
from tqdm import tqdm
src_images = '../data/cleaned'
check_img_loading = False
removed = 0
root = os.path.dirname(src_images)
with open(os.path.join(root, 'image_list.txt'), 'wt') as file:
    for root, dirs, image_names in os.walk(src_images):
        for image_name in tqdm(image_names):
            img_path = os.path.join(root, image_name)
            if check_img_loading:
                try:
                    img = Image.open(img_path)
                    img = img.convert('RGB')
                    if img is None:
                        raise FileNotFoundError
                except:
                    os.remove(img_path)
                    removed += 1
                    continue
            file.write(image_name + '\n')

print(f'removed={removed}')
