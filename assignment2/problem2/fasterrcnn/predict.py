from PIL import Image
import os
from frcnn import FRCNN

if __name__ == "__main__":
    frcnn = FRCNN()
    crop            = False
    count           = False
    dir_origin_path = "img/test2"
    dir_save_path   = "img_out/test2"

    while True:
        img = input('Input image filename:')
        print(os.path.join(dir_origin_path, f"{img}.jpg"))
        try:
            image = Image.open(os.path.join(dir_origin_path, f"{img}.jpg"))
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = frcnn.detect_image(image, crop = crop, count = count)
            r_image.save(os.path.join(dir_save_path, f"{img}_detected.jpg"))
