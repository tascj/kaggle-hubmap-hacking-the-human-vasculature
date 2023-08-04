import json
import os.path as osp
import numpy as np
import pandas as pd
import pycocotools.mask as mask_utils
import mmengine
from sklearn.model_selection import StratifiedKFold

DATA_ROOT = '../data/train'

def load_annotations(ann_file):
    ret = {}
    with open(ann_file) as f:
        for line in f:
            ann = json.loads(line)
            ret[ann['id']] = ann['annotations']
    return ret


def decode_coords(coords):
    rles = mask_utils.frPyObjects([_.flatten().tolist() for _ in np.asarray(coords)], 512, 512)
    rle = mask_utils.merge(rles)
    bbox = mask_utils.toBbox(rle)
    rle['counts'] = rle['counts'].decode()
    return bbox, rle



def df2coco(df, annotations):
    print(df.shape)
    coco = {
        'info': {},
        'categories': [{
            'id': 0,
            'name': 'blood_vessel',
        },{
            'id': 1,
            'name': 'glomerulus',
        },{
            'id': 2,
            'name': 'unsure'
        }]
    }
    img_infos = []
    ann_infos = []
    img_id = 0
    ann_id = 0
    for _, row in df.iterrows():
        _id = row['id']
        img_info = dict(
            id=img_id,
            width=512,
            height=512,
            file_name=f'{_id}.tif',
        )
        anns = annotations[_id]
        for ann in anns:
            if ann['type'] == 'blood_vessel':
                cat_id = 0
            elif ann['type'] == 'glomerulus':
                cat_id = 1
            elif ann['type'] == 'unsure':
                cat_id = 2
            else:
                raise ValueError()
            coords = ann['coordinates']
            xs = np.asarray(coords)
            assert xs.shape[0] == 1
            xmin, ymin = xs[0].min(0)
            xmax, ymax = xs[0].max(0)
            w, h = xmax - xmin, ymax - ymin
            # exit()
            bbox, rle = decode_coords(coords)
            polygon = xs.reshape(1, -1).tolist()
            ann_info = dict(
                id=ann_id,
                image_id=img_id,
                category_id=cat_id,
                iscrowd=0,
                segmentation=polygon,
                area=w * h,
                bbox=[xmin, ymin, w, h],
            )
            ann_infos.append(ann_info)
            ann_id += 1
        img_infos.append(img_info)
        img_id += 1
    coco['images'] = img_infos
    coco['annotations'] = ann_infos
    return coco



annotations = load_annotations('../data/polygons.jsonl')
df = pd.read_csv('../data/tile_meta.csv')
wsi1ds1 = df.query('(dataset == 1) and (source_wsi == 1)')
wsi2ds1 = df.query('(dataset == 1) and (source_wsi == 2)')

# all
trainval = pd.concat([
    wsi1ds1, wsi2ds1
], axis=0)
mmengine.dump(df2coco(trainval, annotations), f'../data/dtrainval.json')

# split by i
val0 = pd.concat([
    wsi1ds1[wsi1ds1['i'] < wsi1ds1['i'].quantile(0.2)],
    wsi2ds1[wsi2ds1['i'] < wsi2ds1['i'].quantile(0.2)],
], axis=0)
train0 = pd.concat([
    wsi1ds1[wsi1ds1['i'] >= wsi1ds1['i'].quantile(0.2)],
    wsi2ds1[wsi2ds1['i'] >= wsi2ds1['i'].quantile(0.2)],
], axis=0)

mmengine.dump(df2coco(train0, annotations), f'../data/dtrain0i.json')
mmengine.dump(df2coco(val0, annotations), f'../data/dval0i.json')


# split by i
val1 = pd.concat([
    wsi1ds1[wsi1ds1['i'] > wsi1ds1['i'].quantile(0.8)],
    wsi2ds1[wsi2ds1['i'] > wsi2ds1['i'].quantile(0.8)],
], axis=0)
train1 = pd.concat([
    wsi1ds1[wsi1ds1['i'] <= wsi1ds1['i'].quantile(0.8)],
    wsi2ds1[wsi2ds1['i'] <= wsi2ds1['i'].quantile(0.8)],
], axis=0)

mmengine.dump(df2coco(train1, annotations), f'../data/dtrain1i.json')
mmengine.dump(df2coco(val1, annotations), f'../data/dval1i.json')

df3 = df.query('(dataset == 2)')
coco3 = df2coco(df3, annotations)
mmengine.dump(coco3, '../data/dtrain_dataset2.json')
