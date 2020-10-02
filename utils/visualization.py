import cv2
import os
import numpy
from PIL import Image

class Visualizer():
    def __init__(self):
        self.class_names = {0:'aeroplane',1:'bicycle',2:'bird',3:'boat',4:'bottle',5:'bus',6:'car',
                            7:'cat',8:'chair',9:'cow',10:'diningtable',11:'dog',12:'horse',13:'motorbike',14:'person',
                            15:'pottedplant',16:'sheep',17:'sofa',18:'train',19:'tvmonitor'}

    def boxes(self,bboxes, img_list):
        # draw the rectangle for every img in img_list according to the bboxes
        # bboxes has N lists, each with top_K lists, each with a turple(class_index,x1,y1,x2,y2,confi)
        bbox_num = bboxes
        imgs = os.listdir(img_list)
        img_num = 0
        for i in imgs:
            if not i.endswith('.jpg'):
                continue
            img_num += 1
        assert img_num == bbox_num, "img_nums is inconsistent with bbox_nums"

        for i in bbox_num:
            img = os.path.join(img_list,i)
            self.draw_box(bboxes[i],img)

        return


    def draw_box(self, bbox, img_path):
        # bbox only for one img
        # bbox[topk(turple)]
        img = cv2.cvtColor(numpy.asarray(img_path),cv2.COLOR_RGB2BGR)
        for i in range(len(bbox)):
            n,x1,y1,x2,y2,confidence = bbox[i]
            confidence = confidence.item()
            class_name = self.class_names[n.item()] + ' {:.2f}'.format(confidence)
            # if confidence < 0.8:
            #     continue
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3)
            cv2.putText(img, class_name, (x1, y1+5),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
        # img_name = img_path.split('/')[-1]
        # img_name = img_name.replace('.jpg','_result.jpg')
        cv2.imwrite('test_out.jpg',img)
        return