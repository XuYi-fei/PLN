import sys
import copy
from torchvision.datasets import VOCDetection
from PIL import  Image
import torch
from torchvision.transforms import transforms
import math
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

class Dataset(VOCDetection):
    def __init__(self,root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 only_img = False):
        super(Dataset, self).__init__(root,year,image_set,download,transform,target_transform,transforms)
        self.only_image = only_img
        self.grid_w = 512 / 14
        self.grid_h = 512 / 14
        self.class2index = {"aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4, "bus": 5, "car": 6, "cat": 7,
                        "chair": 8, "cow": 9, "diningtable": 10, "dog": 11, "horse": 12, "motorbike": 13, "person": 14,
                        "pottedplant": 15, "sheep": 16, "sofa": 17, "train": 18, "tvmonitor": 19}

    def __getitem__(self, index):
        # The initial version only changes the image size
        # Now we want to also apply it to bboxes
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        # transforms is a fuction that is defined customly
        # The following is the expansion of the transforms
        # Change the bbox as the scale which is got from the transform
        # Note: the initial bboxes and sizes in annotations are stored as strings, in transforms, I changed it to int.
        if self.transform is not None:
            origin_img = img
            img = self.transform(img)
            if self.only_image:
                return img
            h_scale = img.height / origin_img.height
            w_scale = img.width / origin_img.width
            object = target['annotation']['object']
            size = target['annotation']['size']

            size['height'] = img.height
            size['width'] = img.width
            # object
            if isinstance(object,list):
                for t in object:
                    t['bndbox']['xmin'] = (float(t['bndbox']['xmin']) * w_scale)
                    t['bndbox']['xmax'] = (float(t['bndbox']['xmax']) * w_scale)
                    t['bndbox']['ymin'] = (float(t['bndbox']['ymin']) * h_scale)
                    t['bndbox']['ymax'] = (float(t['bndbox']['ymax']) * h_scale)
            elif isinstance(object, dict):
                object['bndbox']['xmin'] = (float(object['bndbox']['xmin']) * w_scale)
                object['bndbox']['xmax'] = (float(object['bndbox']['xmax']) * w_scale)
                object['bndbox']['ymin'] = (float(object['bndbox']['ymin']) * h_scale)
                object['bndbox']['ymax'] = (float(object['bndbox']['ymax']) * h_scale)


        elif self.transforms is not None:
            img, target = self.transforms(img, target)

        bbox = self.get_bboxes(target)
        return img, bbox

    def get_bboxes(self,target):
        # get the targets and analyze the bboxes, size, pic_names and classes
        annotation = target['annotation']
        dic = {}
        dic['filename'] = annotation['filename']
        dic['size'] = annotation['size']
        object = annotation['object']
        if isinstance(object,list):
            bboxes = []
            for i in object:
                bbox = {}
                bbox['name'] = i['name']
                bndbox = {}
                bndbox['xmin'] = i['bndbox']['xmin']
                bndbox['xmax'] = i['bndbox']['xmax']
                bndbox['ymin'] = i['bndbox']['ymin']
                bndbox['ymax'] = i['bndbox']['ymax']
                bbox['bndbox'] = bndbox
                bboxes.append(bbox)
            dic['bboxes'] = bboxes
        elif isinstance(object, dict):
            bboxes = []
            bbox = {}
            bbox['name'] = object['name']
            bndbox = {}
            bndbox['xmin'] = object['bndbox']['xmin']
            bndbox['xmax'] = object['bndbox']['xmax']
            bndbox['ymin'] = object['bndbox']['ymin']
            bndbox['ymax'] = object['bndbox']['ymax']
            bbox['bndbox'] = bndbox
            bboxes.append(bbox)
            dic['bboxes'] = bboxes

        dic = self.get_points(dic)

        return dic


    def get_points(self,dic):
        # dic comes from fuction: get_bboxes(), here I decide to get the corner point  and center point and their linking
        for idx, bbox in enumerate(dic['bboxes']):

            # for every bbox, there are four corner points
            p1 = (bbox['bndbox']['xmin'], bbox['bndbox']['ymin'])
            p2 = (bbox['bndbox']['xmin'], bbox['bndbox']['ymax'])
            p3 = (bbox['bndbox']['xmax'], bbox['bndbox']['ymin'])
            p4 = (bbox['bndbox']['xmax'], bbox['bndbox']['ymax'])
            p_center = ((float(p1[0]) + float(p3[0])) / 2, (float(p1[1]) + float(p4[1])) / 2)
            class_index = self.class2index[bbox['name']]
            bbox = (p1,p2,p3,p4,p_center,class_index)
            dic['bboxes'][idx] = bbox
            # each corner point is [[x,y,ix,iy,Lx,Ly],[x,y,ix,iy,Lx,Ly],[x,y,ix,iy,Lx,Ly],[x,y,ix,iy,Lx,Ly],name]
            # corner_point = [self.get_xy(p1),self.get_xy(p2),self.get_xy(p3),self.get_xy(p4), bbox['name']]
            # center point is [[x,y,sx,sy,Lx,Ly],name]
            # and Lx has for elements, because there are four corner points for one center point
        return dic

    def get_xy(self,point):
        # a poing is (bbox['bndbox']['xmin'], bbox['bndbox']['ymin'])
        # return a turple(x,y,ix,iy) or (x,y,sx,sy)
        # Also need to consider links but not implemented so far
        grid_w, grid_h = 512 / 14, 512 / 14
        ix, iy = int(float(point[0]) // grid_w), int(float(point[1]) // grid_h)
        x,y = (float(point[0]) - grid_w * ix) / grid_w, (float(point[1]) - grid_h * iy) / grid_h

        return [x,y,ix,iy]





