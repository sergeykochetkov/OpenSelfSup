import os
import cv2
from PIL import Image
from tqdm import tqdm

src_images = '/home/skochetkov/Documents/isc/data/fb-isc-data-training-images'
check_img_loading = False
save_labels = True
max_imgs = 100000
removed = 0
root = os.path.dirname(src_images)
with open(os.path.join(root, 'image_list.txt'), 'wt') as file:
    for root, dirs, image_names in os.walk(src_images):
        image_names.sort()
        image_names = image_names[:max_imgs]
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
            record = image_name
            if save_labels:
                label = ''.join([i for i in image_name if i.isdigit()])
                label = int(label)
                if label >= max_imgs:
                    break
                record += ' ' + str(label)

            file.write(record + '\n')

print(f'removed={removed}')
