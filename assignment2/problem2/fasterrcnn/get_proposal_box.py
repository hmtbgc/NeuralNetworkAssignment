import numpy as np
from utils.utils import (cvtColor, get_new_img_size, resize_image,
                         preprocess_input)
import torch
from PIL import Image, ImageDraw
from frcnn import FRCNN
import os

def frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes
    
def proposal_box(model, image):
    image_shape = np.array(np.shape(image)[0:2])
    input_shape = get_new_img_size(image_shape[0], image_shape[1])
    image       = cvtColor(image)
    image_data  = resize_image(image, [input_shape[1], input_shape[0]])
    image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    with torch.no_grad():
        images = torch.from_numpy(image_data)
        images = images.cuda()
        roi_cls_locs, roi_scores, rois, _ = model.net(images)
        print(rois.shape)
        
    thickness   = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))
    
    rois = rois.squeeze(0).detach().cpu().numpy()
    rois[..., [0, 2]] = (rois[..., [0, 2]]) / input_shape[1]
    rois[..., [1, 3]] = (rois[..., [1, 3]]) / input_shape[0]
    box_xy, box_wh = (rois[:, 0:2] + rois[:, 2:4]) / 2, rois[:, 2:4] - rois[:, 0:2]
    rois = frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    print(rois.shape)
    
    
    for i in range(rois.shape[0]):

        top, left, bottom, right = rois[i]

        top     = max(0, np.floor(top).astype('int32'))
        left    = max(0, np.floor(left).astype('int32'))
        bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
        right   = min(image.size[0], np.floor(right).astype('int32'))

        draw = ImageDraw.Draw(image)
    

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=128)
        del draw
    
    return image

if __name__ == "__main__":
    model = FRCNN()
    dir_origin_path = "img/test1"
    dir_save_path   = "img_out/test1"
    while True:
        img = input('Input image filename:')
        image = Image.open(os.path.join(dir_origin_path, f"{img}.jpg"))
        image_with_proposal_box = proposal_box(model, image)
        image_with_proposal_box.save(os.path.join(dir_save_path, f"{img}_proposal_box.jpg"))
    
        