import xml.etree.ElementTree as ET
import collections
import glob
from tqdm import tqdm
import json
import random
import sys
import os
import numpy as np
import cv2

CLASS_DICT = collections.OrderedDict({
'mask':1,
'head':2, 
'back':3,
'mid_mask':4})


def read_xml(xml_file, img_id):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    file_name = root.find('filename').text
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    segmented = int(root.find('segmented').text)
    objects = root.findall('object')
    annos = []
    for i, obj in enumerate(objects):
        cls_id = CLASS_DICT[obj.find('name').text]
        img_id = obj.find('name').text
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)
        box_w = xmax - xmin
        box_h = ymax - ymin
        pose = obj.find('pose').text
        truncated = int(obj.find('truncated').text)
        difficult = int(obj.find('difficult').text)
        annos.append({'image_id':img_id,
                        'bbox': [xmin, ymin, box_w, box_h],
                        'category_id': cls_id,
                        'pose': pose,
                        'truncated': truncated,
                        'difficult': difficult,
                        'iscrowd':0,
                        'segmentation': [],
                        'area': int(box_w * box_h),
                        'id': i})

    return file_name, width, height, segmented, annos

print('constructing dataset')
images = []
annotations = []
categories = []
img_id = 0
raw_data_path = sys.argv[1]
anno_save_path = sys.argv[1] + '/my_dataset/annotations'
if not os.path.exists(anno_save_path):
    os.makedirs(anno_save_path)
anno_paths = glob.glob(raw_data_path + '/*/*.xml')
# img_paths = glob.glob(raw_data_path + '/*.jpg')
# img_paths.extend(glob.glob(raw_data_path + '/*.png'))
for k in list(CLASS_DICT.keys()):
    categories.append({"id": CLASS_DICT[k], "name":k})
m_list = []
s_list = []
# for ip in tqdm(anno_paths):
for ip in anno_paths:
    file_name, w, h, segmented, annos = read_xml(ip, img_id)
    file_name = ip.split('/')[-2] + '/' + file_name
    images.append({'file_name':file_name,
                    'id':img_id,
                    'height':h,
                    'width':w})
    annotations.extend(annos)
    img_id += 1
#     import pdb;pdb.set_trace()
    img = cv2.imread(raw_data_path + '/' + file_name)
    img = img / 255.0
    m, s = cv2.meanStdDev(img)
    m_list.append(m.reshape((3,)))
    s_list.append(s.reshape((3,)))
m_array = np.array(m_list)
s_array = np.array(s_list)
m = m_array.mean(axis=0, keepdims=True)
s = s_array.mean(axis=0, keepdims=True)
img_info = {}
img_info['mean'] = m[0][::-1].tolist()
img_info['std'] = s[0][::-1].tolist()
with open(anno_save_path+'/img_info.json', 'w') as json_f3:          
    json.dump(img_info, json_f3)
all_anns = {"images": images, "annotations":annotations, "categories":categories}
# #store all annotations
# with open(anno_save_path + 'trainval.json', 'w') as json_f3:
#     json.dump(all_anns, json_f3)

print('spliting dataset')
length = len(all_anns["images"])
rd_idx = list(range(length))
random.shuffle(rd_idx)
# print(rd_idx)
rate = 0.8
train_idx = rd_idx[:int(length*0.8)]
val_idx = rd_idx[int(length*0.8):]
imgs = all_anns["images"]
annos = all_anns["annotations"]
train_images = []
val_images = []
train_annotations = []
val_annotations = []
for i in train_idx:
    train_images.append(imgs[i])
    train_annotations.append(annos[i])
for i in val_idx:
    val_images.append(imgs[i])
    val_annotations.append(annos[i])

train_cat = all_anns["categories"]
val_cat = all_anns["categories"]

train_anns = {"images": train_images, "annotations":train_annotations, "categories":train_cat} 
val_anns = {"images": val_images, "annotations":val_annotations, "categories":val_cat}
with open(anno_save_path+'/train.json', 'w') as json_f4:          
    json.dump(train_anns, json_f4)       
with open(anno_save_path + '/val.json', 'w') as json_f5:          
    json.dump(val_anns, json_f5)