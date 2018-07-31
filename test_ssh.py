from __future__ import print_function
import argparse
import sys
import os
import time
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import cv2
from rcnn.logger import logger
from rcnn.config import config, default, generate_config
#from rcnn.tools.test_rcnn import test_rcnn
#from rcnn.tools.test_rpn import test_rpn
from rcnn.processing.bbox_transform import nonlinear_pred, clip_boxes
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper
from rcnn.processing.bbox_transform import bbox_overlaps
from rcnn.dataset import widerface

class SSHDetector:
  def __init__(self, prefix, epoch, ctx_id=0, test_mode=False):
    self.ctx_id = ctx_id
    self.ctx = mx.gpu(self.ctx_id)
    self.fpn_keys = []
    fpn_stride = []
    fpn_base_size = []
    self._feat_stride_fpn = [32, 16, 8]

    for s in self._feat_stride_fpn:
        self.fpn_keys.append('stride%s'%s)
        fpn_stride.append(int(s))
        fpn_base_size.append(16)

    self._scales = np.array([32,16,8,4,2,1])
    self._ratios = np.array([1.0]*len(self._feat_stride_fpn))
    #self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(base_size=fpn_base_size, scales=self._scales, ratios=self._ratios)))
    self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn()))
    self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
    self._rpn_pre_nms_top_n = 1000
    #self._rpn_post_nms_top_n = rpn_post_nms_top_n
    #self.score_threshold = 0.05
    self.nms_threshold = 0.3
    self._bbox_pred = nonlinear_pred
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    self.nms = gpu_nms_wrapper(self.nms_threshold, self.ctx_id)
    self.pixel_means = np.array([103.939, 116.779, 123.68]) #BGR
    self.pixel_means = config.PIXEL_MEANS
    print('means', self.pixel_means)

    if not test_mode:
      image_size = (640, 640)
      self.model = mx.mod.Module(symbol=sym, context=self.ctx, label_names = None)
      self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))], for_training=False)
      self.model.set_params(arg_params, aux_params)
    else:
      from rcnn.core.module import MutableModule
      image_size = (2400, 2400)
      data_shape = [('data', (1,3,image_size[0], image_size[1]))]
      self.model = MutableModule(symbol=sym, data_names=['data'], label_names=None,
                                context=self.ctx, max_data_shapes=data_shape)
      self.model.bind(data_shape, None, for_training=False)
      self.model.set_params(arg_params, aux_params)


  def detect(self, img, threshold=0.05, scales=[1.0]):
    proposals_list = []
    scores_list = []

    for im_scale in scales:

      if im_scale!=1.0:
        im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
      else:
        im = img
      im = im.astype(np.float32)
      #self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))], for_training=False)
      im_info = [im.shape[0], im.shape[1], im_scale]
      im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
      for i in range(3):
          im_tensor[0, i, :, :] = im[:, :, 2 - i] - self.pixel_means[2 - i]
      data = nd.array(im_tensor)
      db = mx.io.DataBatch(data=(data,), provide_data=[('data', data.shape)])
      self.model.forward(db, is_train=False)
      net_out = self.model.get_outputs()
      pre_nms_topN = self._rpn_pre_nms_top_n
      #post_nms_topN = self._rpn_post_nms_top_n
      #min_size_dict = self._rpn_min_size_fpn

      for s in self._feat_stride_fpn:
          if len(scales)>1 and s==32 and im_scale==scales[-1]:
            continue
          _key = 'stride%s'%s
          stride = int(s)
          idx = 0
          if s==16:
            idx=2
          elif s==8:
            idx=4
          print('getting', im_scale, stride, idx, len(net_out), data.shape, file=sys.stderr)
          scores = net_out[idx].asnumpy()
          #print(scores.shape)
          idx+=1
          #print('scores',stride, scores.shape, file=sys.stderr)
          scores = scores[:, self._num_anchors['stride%s'%s]:, :, :]
          bbox_deltas = net_out[idx].asnumpy()

          #if DEBUG:
          #    print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
          #    print 'scale: {}'.format(im_info[2])

          _height, _width = int(im_info[0] / stride), int(im_info[1] / stride)
          height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

          A = self._num_anchors['stride%s'%s]
          K = height * width

          anchors = anchors_plane(height, width, stride, self._anchors_fpn['stride%s'%s].astype(np.float32))
          #print((height, width), (_height, _width), anchors.shape, bbox_deltas.shape, scores.shape, file=sys.stderr)
          anchors = anchors.reshape((K * A, 4))

          #print('pre', bbox_deltas.shape, height, width)
          bbox_deltas = self._clip_pad(bbox_deltas, (height, width))
          #print('after', bbox_deltas.shape, height, width)
          bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

          scores = self._clip_pad(scores, (height, width))
          scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

          #print(anchors.shape, bbox_deltas.shape, A, K, file=sys.stderr)
          proposals = self._bbox_pred(anchors, bbox_deltas)
          #proposals = anchors

          proposals = clip_boxes(proposals, im_info[:2])

          #keep = self._filter_boxes(proposals, min_size_dict['stride%s'%s] * im_info[2])
          #proposals = proposals[keep, :]
          #scores = scores[keep]
          #print('333', proposals.shape)

          scores_ravel = scores.ravel()
          order = scores_ravel.argsort()[::-1]
          if pre_nms_topN > 0:
              order = order[:pre_nms_topN]
          proposals = proposals[order, :]
          scores = scores[order]

          proposals /= im_scale

          proposals_list.append(proposals)
          scores_list.append(scores)

    proposals = np.vstack(proposals_list)
    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]
    #if config.TEST.SCORE_THRESH>0.0:
    #  _count = np.sum(scores_ravel>config.TEST.SCORE_THRESH)
    #  order = order[:_count]
    #if pre_nms_topN > 0:
    #    order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    det = np.hstack((proposals, scores)).astype(np.float32)

    #if np.shape(det)[0] == 0:
    #    print("Something wrong with the input image(resolution is too low?), generate fake proposals for it.")
    #    proposals = np.array([[1.0, 1.0, 2.0, 2.0]]*post_nms_topN, dtype=np.float32)
    #    scores = np.array([[0.9]]*post_nms_topN, dtype=np.float32)
    #    det = np.array([[1.0, 1.0, 2.0, 2.0, 0.9]]*post_nms_topN, dtype=np.float32)

    
    if self.nms_threshold<1.0:
      keep = self.nms(det)
      det = det[keep, :]
    if threshold>0.0:
      keep = np.where(det[:, 4] >= threshold)[0]
      det = det[keep, :]
    return det

  @staticmethod
  def _filter_boxes(boxes, min_size):
      """ Remove all boxes with any side smaller than min_size """
      ws = boxes[:, 2] - boxes[:, 0] + 1
      hs = boxes[:, 3] - boxes[:, 1] + 1
      keep = np.where((ws >= min_size) & (hs >= min_size))[0]
      return keep

  @staticmethod
  def _clip_pad(tensor, pad_shape):
      """
      Clip boxes of the pad area.
      :param tensor: [n, c, H, W]
      :param pad_shape: [h, w]
      :return: [n, c, h, w]
      """
      H, W = tensor.shape[2:]
      h, w = pad_shape

      if h < H or w < W:
        tensor = tensor[:, :, :h, :w].copy()

      return tensor

def parse_args():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    parser.add_argument('--image_set', help='image_set name', default=default.test_image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    # testing
    parser.add_argument('--prefix', help='model to test with', default=default.e2e_prefix, type=str)
    parser.add_argument('--epoch', help='model to test with', default=0, type=int)
    parser.add_argument('--gpu', help='GPU device to test with', default=7, type=int)
    parser.add_argument('--output', help='output folder', default=os.path.join(default.root_path, 'output'), type=str)
    parser.add_argument('--pyramid', help='enable pyramid test', action='store_true')
    # rcnn
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=0.05, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--has_rpn', help='generate proposals on the fly', action='store_true', default=True)
    parser.add_argument('--proposal', help='can be ss for selective search or rpn', default='rpn', type=str)
    args = parser.parse_args()
    return args

detector = None
args = None

def get_boxes(roi, pyramid):
  im = cv2.imread(roi['image'])
  if not pyramid:
    target_size = 1200
    max_size = 1600
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    scales = [im_scale]
  else:
    TEST_SCALES = [500, 800, 1200, 1600]
    target_size = 800
    max_size = 1200
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    scales = [float(scale)/target_size*im_scale for scale in TEST_SCALES]
  boxes = detector.detect(im, threshold=args.thresh, scales = scales)
  return boxes


def test(args):
  print('test with', args)
  global detector
  output_folder = args.output
  if not os.path.exists(output_folder):
    os.mkdir(output_folder)
  detector = SSHDetector(args.prefix, args.epoch, args.gpu, test_mode=True)
  imdb = eval(args.dataset)(args.image_set, args.root_path, args.dataset_path)
  roidb = imdb.gt_roidb()
  gt_overlaps = np.zeros(0)
  overall = [0.0, 0.0]
  gt_max = np.array( (0.0, 0.0) )
  num_pos = 0

  for i in xrange(len(roidb)):
    if i%10==0:
      print('processing', i, file=sys.stderr)
    roi = roidb[i]
    boxes = get_boxes(roi, args.pyramid)
    gt_boxes = roidb[i]['boxes'].copy()
    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    num_pos += gt_boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))
    #print(im_info, gt_boxes.shape, boxes.shape, overlaps.shape, file=sys.stderr)

    _gt_overlaps = np.zeros((gt_boxes.shape[0]))

    if boxes.shape[0]>0:
      _gt_overlaps = overlaps.max(axis=0)
      #print('max_overlaps', _gt_overlaps, file=sys.stderr)
      for j in range(len(_gt_overlaps)):
        if _gt_overlaps[j]>config.TEST.IOU_THRESH:
          continue
        print(j, 'failed', gt_boxes[j],  'max_overlap:', _gt_overlaps[j], file=sys.stderr)

      # append recorded IoU coverage level
      found = (_gt_overlaps > config.TEST.IOU_THRESH).sum()
      _recall = found / float(gt_boxes.shape[0])
      print('recall', _recall, gt_boxes.shape[0], boxes.shape[0], gt_areas, file=sys.stderr)
      overall[0]+=found
      overall[1]+=gt_boxes.shape[0]
      #gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))
      #_recall = (gt_overlaps >= threshold).sum() / float(num_pos)
      _recall = float(overall[0])/overall[1]
      print('recall_all', _recall, file=sys.stderr)


    _vec = roidb[i]['image'].split('/')
    out_dir = os.path.join(output_folder, _vec[-2])
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_file = os.path.join(out_dir, _vec[-1].replace('jpg', 'txt'))
    with open(out_file, 'w') as f:
      name = '/'.join(roidb[i]['image'].split('/')[-2:])
      f.write("%s\n"%(name))
      f.write("%d\n"%(boxes.shape[0]))
      for b in range(boxes.shape[0]):
        box = boxes[b]
        f.write("%d %d %d %d %g \n"%(box[0], box[1], box[2]-box[0], box[3]-box[1], box[4]))

def main():
    global args
    args = parse_args()
    logger.info('Called with argument: %s' % args)
    test(args)

if __name__ == '__main__':
    main()

