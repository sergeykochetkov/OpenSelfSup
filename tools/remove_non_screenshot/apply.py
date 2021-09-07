import os
import shutil
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from PIL import Image
import torch

from train import CHECKPOINT_PATH, get_transforms, get_model

if __name__ == "__main__":
    src_path = '/home/skochetkov/Documents/OpenSelfSup/data/raw'
    dst_path = '/home/skochetkov/Documents/OpenSelfSup/data/cleaned'
    dst_path_out = '/home/skochetkov/Documents/OpenSelfSup/data/out'

    prob_thr = 0.7

    model0 = torch.load(CHECKPOINT_PATH)
    model0.eval()
    model0.to('cuda:0')
    model1 = torch.load(CHECKPOINT_PATH)
    model1.eval()
    model1.to('cuda:1')


    def _job(i, root, files):
        i = i % 2
        device = f'cuda:{i}'
        model = model0 if i == 0 else model1

        for file in tqdm(files):
            img_path = os.path.join(root, file)
            try:
                img = Image.open(img_path)
                img = img.convert('RGB')
                tensor_img = test_transform(img)
                tensor_img = torch.unsqueeze(tensor_img, dim=0)
                tensor_img = tensor_img.to(device)
                prediction = model(tensor_img)
                probs = torch.exp(prediction).detach().cpu().numpy()
                if probs[0, 1] > prob_thr:
                    shutil.copyfile(img_path, os.path.join(dst_path, file))
                else:
                    shutil.copyfile(img_path, os.path.join(dst_path_out, file))
            except:
                pass


    _, test_transform = get_transforms()

    os.makedirs(dst_path, exist_ok=True)
    os.makedirs(dst_path_out, exist_ok=True)

    Parallel(n_jobs=2)(delayed(_job)(i, root, files) for i, (root, dirs, files) in enumerate(os.walk(src_path)))
