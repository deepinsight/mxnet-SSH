"""
ICDAR15 Database
"""
import scipy.io as sio
import cv2
import os
import numpy as np
import cPickle
import PIL.Image as Image
from imdb import IMDB
from ..processing.bbox_transform import bbox_overlaps

#class Cityscape(IMDB):
class ICDAR15(IMDB):
    def __init__(self, image_set, root_path, dataset_path):
        """
        fill basic information to initialize imdb
        :param image_set: train or val or trainval or test
        :param root_path: 'cache' and 'rpn_data'
        :param dataset_path: data and results
        :return: imdb object
        """
        year, image_set = image_set.split('_')
        super(ICDAR15, self).__init__('icdar15', image_set, root_path, dataset_path)
        self.image_set = image_set
        self.root_path = root_path
        self.data_path = dataset_path

        #self.classes = ['__background__', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
        #self.class_id = [0, 24, 25, 26, 27, 28, 31, 32, 33]
        self.classes = ['__background__', 'text']
        self.class_id = [0, 1]
        self.num_classes = len(self.classes)
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        if not os.path.exists(image_file):
            image_file = os.path.join(self.data_path, 'img', index + '.jpg')
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'ins_id', 'ins_seg', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                roidb = cPickle.load(f)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        #gt_roidb = self.load_cityscape_annotations()

        gt_roidb = [self.load_cityscape_annotations(index) for index in self.image_set_index]

        with open(cache_file, 'wb') as f:
            cPickle.dump(gt_roidb, f, cPickle.HIGHEST_PROTOCOL)

        return gt_roidb

    def load_from_seg(self, ins_seg_path):
        seg_gt = os.path.join(self.data_path, ins_seg_path)
        print seg_gt
        assert os.path.exists(seg_gt), 'Path does not exist: {}'.format(seg_gt)
        im = Image.open(seg_gt)
        pixel = list(im.getdata())
        pixel = np.array(pixel).reshape([im.size[1], im.size[0]])
        print im.size
        boxes = []
        gt_classes = []
        ins_id = []
        gt_overlaps = []
        for c in range(1, len(self.class_id)):
            px = np.where((pixel >= self.class_id[c] * 1000) & (pixel < (self.class_id[c] + 1) * 1000))
            if len(px[0]) == 0:
                continue
            ids = np.unique(pixel[px])
            for id in ids:
                px = np.where(pixel == id)
                x_min = np.min(px[1])
                y_min = np.min(px[0])
                x_max = np.max(px[1])
                y_max = np.max(px[0])
                if x_max - x_min <= 1 or y_max - y_min <= 1:
                    continue
                boxes.append([x_min, y_min, x_max, y_max])
                gt_classes.append(c)
                ins_id.append(id % 1000)
                overlaps = np.zeros(self.num_classes)
                overlaps[c] = 1
                gt_overlaps.append(overlaps)
        return np.asarray(boxes), np.asarray(gt_classes), np.asarray(ins_id), seg_gt, np.asarray(gt_overlaps)

    def load_cityscape_annotations(self, index):
        sds_rec = dict()
        sds_rec['image'] = self.image_path_from_index(index)
        size = cv2.imread(sds_rec['image']).shape
        sds_rec['height'] = size[0]
        sds_rec['width'] = size[1]

        # class level segmentation
        seg_cls_name = os.path.join(self.data_path, 'cls', index + '.mat')
        seg_cls_mat = sio.loadmat(seg_cls_name)
        #seg_cls_data = seg_cls_mat['GTcls']['Segmentation'][0][0]
        
        seg_cls_data = seg_cls_mat['Segmentation']
        # instance level segmentation
        seg_obj_name = os.path.join(self.data_path, 'inst', index + '.mat')
        seg_obj_mat = sio.loadmat(seg_obj_name)
        #seg_obj_data = seg_obj_mat['GTinst']['Segmentation'][0][0]
        seg_obj_data = seg_obj_mat['Segmentation']
        unique_inst = np.unique(seg_obj_data)
        background_ind = np.where(unique_inst == 0)[0]
        unique_inst = np.delete(unique_inst, background_ind)
        border_inds = np.where(unique_inst == 255)[0]
        unique_inst = np.delete(unique_inst, border_inds)

        num_objs = len(unique_inst)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros(num_objs, dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        gt_masks = np.zeros((num_objs, size[0], size[1]))
        ins_id = []
        seg_gt = seg_obj_name
        for idx, inst_id in enumerate(unique_inst):
            [r, c] = np.where(seg_obj_data == inst_id)
            x1 = np.min(c)
            x2 = np.max(c)
            y1 = np.min(r)
            y2 = np.max(r)
            cur_gt_mask = (seg_obj_data == inst_id)
            cur_gt_mask_cls = seg_cls_data[cur_gt_mask]
            assert np.unique(cur_gt_mask_cls).shape[0] == 1
            cur_inst_cls = int(np.unique(cur_gt_mask_cls)[0])

            boxes[idx, :] = [x1, y1, x2, y2]
            gt_classes[idx] = cur_inst_cls
            gt_masks[idx, :, :] = cur_gt_mask
            #print(idx)
            #print(cur_inst_cls)
            overlaps[idx, cur_inst_cls] = 1.0
            ins_id.append(idx)

        sds_rec.update({
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'ins_id' : ins_id,
            'ins_seg' : seg_gt,
            'max_classes': overlaps.argmax(axis=1),
            'max_overlaps': overlaps.max(axis=1),
            #'cache_seg_inst': self.mask_path_from_index(index, gt_masks),
            'flipped': False
        })
        return sds_rec

    def append_flipped_images(self, roidb):
        """
        append flipped images to an roidb
        flip boxes coordinates, images will be actually flipped when loading into network
        :param roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        print 'append flipped images to roidb'
        assert self.num_images == len(roidb)
        for i in range(self.num_images):
            roi_rec = roidb[i]
            boxes = roi_rec['boxes'].copy()
            if boxes.shape[0] != 0:
                oldx1 = boxes[:, 0].copy()
                oldx2 = boxes[:, 2].copy()
                boxes[:, 0] = roi_rec['width'] - oldx2 - 1
                boxes[:, 2] = roi_rec['width'] - oldx1 - 1
                assert (boxes[:, 2] >= boxes[:, 0]).all(),\
                    'img_name %s, width %d\n' % (roi_rec['image'], roi_rec['width']) + \
                    np.array_str(roi_rec['boxes'], precision=3, suppress_small=True)
            entry = {'image': roi_rec['image'],
                     'height': roi_rec['height'],
                     'width': roi_rec['width'],
                     'boxes': boxes,
                     'gt_classes': roidb[i]['gt_classes'],
                     'gt_overlaps': roidb[i]['gt_overlaps'],
                     'max_classes': roidb[i]['max_classes'],
                     'max_overlaps': roidb[i]['max_overlaps'],
                     'ins_seg': roidb[i]['ins_seg'],
                     'ins_id': roidb[i]['ins_id'],
                     'flipped': True}
            roidb.append(entry)

        self.image_set_index *= 2
        return roidb

    def evaluate_mask(self, results_pack):
        for result_rec in results_pack['results_list']:
            image_path = result_rec['image']
            im_info = result_rec['im_info']
            detections = result_rec['boxes']
            seg_masks = result_rec['masks']

            filename = image_path.split("/")[-1]
            filename = filename.replace('.png', '')

            result_path = 'data/cityscape/results/pred/'

            print 'writing results for: ', filename
            result_txt = os.path.join(result_path, filename)
            result_txt = result_txt + '.txt'
            count = 0
            f = open(result_txt, 'w')

            for j, labelID in enumerate(self.class_id):
                if labelID == 0:
                    continue
                dets = detections[j]
                masks = seg_masks[j]
                for i in range(len(dets)):
                    bbox = dets[i, :4]
                    score = dets[i, -1]
                    bbox = map(int, bbox)
                    mask_image = np.zeros((int(im_info[0, 0]), int(im_info[0, 1])))
                    mask = masks[i, :, :]
                    mask = cv2.resize(mask, (bbox[2] - bbox[0], (bbox[3] - bbox[1])), interpolation=cv2.INTER_LINEAR)
                    mask[mask > 0.5] = 200
                    mask[mask <= 0.5] = 0
                    mask_image[bbox[1]: bbox[3], bbox[0]: bbox[2]] = mask
                    cv2.imwrite(os.path.join(result_path, filename) + '_' + str(count) + '.png', mask_image)
                    f.write('{:s} {:s} {:.8f}\n'.format(filename + '_' + str(count) + '.png', str(labelID), score))
                    count += 1
            f.flush()
            f.close()

    def evaluate_detections(self, detections):
        raise NotImplementedError

    def create_roidb_from_box_list(self, box_list, gt_roidb):
        """
        given ground truth, prepare roidb
        :param box_list: [image_index] ndarray of [box_index][x1, x2, y1, y2]
        :param gt_roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        assert len(box_list) == self.num_images, 'number of boxes matrix must match number of images'
        roidb = []
        for i in range(self.num_images):
            roi_rec = dict()
            roi_rec['image'] = gt_roidb[i]['image']
            roi_rec['height'] = gt_roidb[i]['height']
            roi_rec['width'] = gt_roidb[i]['width']

            boxes = box_list[i]
            if boxes.shape[1] == 5:
                boxes = boxes[:, :4]
            num_boxes = boxes.shape[0]
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)
            if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
                gt_boxes = gt_roidb[i]['boxes']
                gt_classes = gt_roidb[i]['gt_classes']
                # n boxes and k gt_boxes => n * k overlap
                gt_overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))
                # for each box in n boxes, select only maximum overlap (must be greater than zero)
                argmaxes = gt_overlaps.argmax(axis=1)
                maxes = gt_overlaps.max(axis=1)
                I = np.where(maxes > 0)[0]
                overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            roi_rec.update({'boxes': boxes,
                            'gt_classes': np.zeros((num_boxes,), dtype=np.int32),
                            'gt_overlaps': overlaps,
                            'max_classes': overlaps.argmax(axis=1),
                            'max_overlaps': overlaps.max(axis=1),
                            'ins_seg': gt_roidb[i]['ins_seg'],
                            'ins_id': gt_roidb[i]['ins_id'],
                            'flipped': False})

            # background roi => background class
            zero_indexes = np.where(roi_rec['max_overlaps'] == 0)[0]
            assert all(roi_rec['max_classes'][zero_indexes] == 0)
            # foreground roi => foreground class
            nonzero_indexes = np.where(roi_rec['max_overlaps'] > 0)[0]
            assert all(roi_rec['max_classes'][nonzero_indexes] != 0)

            roidb.append(roi_rec)

        return roidb

    def rpn_roidb(self, gt_roidb, append_gt=False):
        """
        get rpn roidb and ground truth roidb
        :param gt_roidb: ground truth roidb
        :param append_gt: append ground truth
        :return: roidb of rpn
        """
        if append_gt:
            print 'appending ground truth annotations'
            rpn_roidb = self.load_rpn_roidb(gt_roidb)
            roidb = IMDB.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self.load_rpn_roidb(gt_roidb)
        return roidb
