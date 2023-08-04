import mmengine
import pycocotools.mask as mask_utils
from pycocotools.coco import COCO


def decode_mask(ann):
    return mask_utils.decode(mask_utils.frPyObjects(ann['segmentation'], 512, 512))

def mask_iou(a, b):
    inter = (a == 1) & (b == 1)
    union = (a == 1) | (b == 1)
    return inter.sum() / union.sum()

coco = COCO('../data/dtrain_dataset2.json')
count = 0
valid_ann_ids = []
for img_id in coco.getImgIds():
    anns = coco.loadAnns(coco.getAnnIds(img_id))
    masks = [decode_mask(ann) for ann in anns]
    

    ious = {}
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            iou = mask_iou(masks[i], masks[j])
            ious[(i, j)] = iou
    
    valid_inds = set()
    for candidate in range(len(masks)):
        if len(valid_inds) == 0:
            valid_inds.add(candidate)
        else:
            add_new = True
            for valid_ind in valid_inds:
                if ious[tuple(sorted([valid_ind, candidate]))] == 1.0:
                    print('duplicate')
                    add_new = False
                    break
            if add_new:
                valid_inds.add(candidate)
    print(len(masks), len(valid_inds))
    if len(masks) != len(valid_inds):
        count += len(masks) - len(valid_inds)

    for valid_ind in valid_inds:
        valid_ann_ids.append(anns[valid_ind]['id'])


valid_ann_ids = set(valid_ann_ids)

d = mmengine.load('../data/dtrain_dataset2.json')
annotations = [
    ann for ann in d['annotations']
    if ann['id'] in valid_ann_ids
]
d['annotations'] = annotations
mmengine.dump(d, '../data/dtrain_dataset2_drop.json')
