import numpy as np
from torchvision.datasets import CIFAR100
from PIL import Image
import os

def mixup(img1, img2, alpha):
    img1 = np.array(img1)
    img2 = np.array(img2)
    lam = np.random.beta(alpha, alpha)
    img = img1 * lam + img2 * (1 - lam)
    img = img.astype(np.uint8)
    return Image.fromarray(img)

def cutout(img, length, n_holes=1):
    img = np.array(img)
    h, w = img.shape[0], img.shape[1]
    mask = np.ones((h, w), np.float32)
    for i in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
    
    mask = np.expand_dims(mask, axis=-1)
    img = (img * mask).astype(np.uint8)
    return Image.fromarray(img)

def cutmix(img1, img2, alpha, length):
    img1 = np.array(img1)
    img2 = np.array(img2)
    lam = np.random.beta(alpha, alpha)
    h, w = img1.shape[0], img1.shape[1]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    img1[bbx1:bbx2, bby1:bby2] = img2[bbx1:bbx2, bby1:bby2]
    img1 = img1.astype(np.uint8)
    return Image.fromarray(img1)


alpha = 0.5
length = 32
pic_root = "./test_pic"
out_root = "./visualization"

img1 = Image.open(os.path.join(pic_root, "bird_re.jpg"))
img2 = Image.open(os.path.join(pic_root, "cat_re.jpg"))
img3 = Image.open(os.path.join(pic_root, "dog_re.jpg"))

img1_cutout = cutout(img1, length)
img1_cutout.save(os.path.join(out_root, "bird_cutout.png"))
img2_cutout = cutout(img2, length)
img2_cutout.save(os.path.join(out_root, "cat_cutout.png"))
img3_cutout = cutout(img3, length)
img3_cutout.save(os.path.join(out_root, "dog_cutout.png"))

img1_mixup_img2 = mixup(img1, img2, alpha)
img1_mixup_img2.save(os.path.join(out_root, "bird_mixup_cat.png"))
img2_mixup_img3 = mixup(img2, img3, alpha)
img2_mixup_img3.save(os.path.join(out_root, "cat_mixup_dog.png"))
img3_mixup_img1 = mixup(img3, img1, alpha)
img3_mixup_img1.save(os.path.join(out_root, "dog_mixup_bird.png"))

img1_cutmix_img2 = cutmix(img1, img2, alpha, length)
img1_cutmix_img2.save(os.path.join(out_root, "bird_cutmix_cat.png"))
img2_cutmix_img3 = cutmix(img2, img3, alpha, length)
img2_cutmix_img3.save(os.path.join(out_root, "cat_cutmix_dog.png"))
img3_cutmix_img1 = cutmix(img3, img1, alpha, length)
img3_cutmix_img1.save(os.path.join(out_root, "dog_cutmix_bird.png"))




