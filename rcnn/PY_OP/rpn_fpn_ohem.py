
from __future__ import print_function
import sys
import mxnet as mx
import numpy as np
from distutils.util import strtobool
from ..config import config

CALLING_COUNT=0
COUNT_THRESH_FG=0
COUNT_THRESH_BG=0
STAT = {0:0, 8:0, 16:0, 32:0}
ACC = {0:0, 8:0, 16:0, 32:0}

class RPNFPNOHEMOperator(mx.operator.CustomOp):
    def __init__(self, stride=0):
        super(RPNFPNOHEMOperator, self).__init__()
        self.stride = int(stride)

    def forward(self, is_train, req, in_data, out_data, aux):
        global STAT
        global ACC

        cls_score    = in_data[0].asnumpy() #BS, 2, ANCHORS
        bbox_weight = in_data[1].asnumpy() #BS, 4*SCALES, featuremapsize
        labels_raw       = in_data[2].asnumpy() # BS, ANCHORS

        A = config.NUM_ANCHORS

        #assert labels.shape[0]==1
        #assert cls_score.shape[0]==1
        #assert bbox_weight.shape[0]==1
        #print('shape', cls_score.shape, labels.shape, file=sys.stderr)
        #print('bbox_weight 0', bbox_weight.shape, file=sys.stderr)
        #bbox_weight = np.zeros( (labels_raw.shape[0], labels_raw.shape[1], 4), dtype=np.float32)
        for ibatch in xrange(labels_raw.shape[0]):
          _bbox_weight = np.zeros( (labels_raw.shape[1], 4), dtype=np.float32)
          labels = labels_raw[ibatch]
          fg_score = cls_score[ibatch,1,:] - cls_score[ibatch,0,:]



          num_fg = int(config.TRAIN.RPN_FG_FRACTION * config.TRAIN.RPN_BATCH_SIZE)
          fg_inds = np.where(labels == 1)[0]
          origin_num_fg = len(fg_inds)
          if len(fg_inds) > num_fg:
            if CALLING_COUNT<COUNT_THRESH_FG:
              disable_inds = np.random.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
              labels[disable_inds] = -1
            else:
              pos_ohem_scores = fg_score[fg_inds]
              order_pos_ohem_scores = pos_ohem_scores.ravel().argsort()
              sampled_inds = fg_inds[order_pos_ohem_scores[:num_fg]]
              labels[fg_inds] = -1
              labels[sampled_inds] = 1

          n_fg = np.sum(labels == 1)
          fg_inds = np.where(labels == 1)[0]
          STAT[0]+=1
          STAT[self.stride] += n_fg
          ACC[self.stride] += np.sum(fg_score[fg_inds]>=0)
          if STAT[0]%9600==0:
            S = {0: STAT[0]}
            for k in STAT:
              if k==0:
                continue
              acc = float(ACC[k])/STAT[k]
              S[k] = (STAT[k], ACC[k], acc)
            print('STAT ', S, file=sys.stderr)
            for k in STAT:
              STAT[k]=0
              ACC[k] = 0
            #print('ohem_calling_count', CALLING_COUNT, STAT, file=sys.stderr)
          num_bg = config.TRAIN.RPN_BATCH_SIZE - n_fg
          bg_inds = np.where(labels == 0)[0]
          origin_num_bg = len(bg_inds)
          if num_bg==0:
            labels[bg_inds] = -1
          elif len(bg_inds) > num_bg:
            # sort ohem scores
            if CALLING_COUNT<COUNT_THRESH_BG:
              disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
              labels[disable_inds] = -1
            else:
              neg_ohem_scores = fg_score[bg_inds]
              order_neg_ohem_scores = neg_ohem_scores.ravel().argsort()[::-1]
              sampled_inds = bg_inds[order_neg_ohem_scores[:num_bg]]
              #print('sampled_inds_bg', sampled_inds, file=sys.stderr)
              labels[bg_inds] = -1
              labels[sampled_inds] = 0

          if n_fg>0:
            order0_labels = labels.reshape( (1, A, -1) ).transpose( (0, 2, 1) ).reshape( (-1,) )
            bbox_fg_inds = np.where(order0_labels == 1)[0]
            #print('bbox_fg_inds, order0 ', bbox_fg_inds, file=sys.stderr)
            _bbox_weight[bbox_fg_inds,:] = np.array(config.TRAIN.RPN_BBOX_WEIGHTS)
          _bbox_weight = _bbox_weight.reshape((1, -1, A * 4)).transpose((0,2,1))
          bbox_weight[ibatch] = _bbox_weight


          #labels = labels[np.newaxis,:]

        labels_ohem = mx.nd.array(labels_raw)
        bbox_weights_ohem = mx.nd.array(bbox_weight)

        for ind, val in enumerate([labels_ohem, bbox_weights_ohem]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register('rpn_fpn_ohem')
class RPNFPNOHEMProp(mx.operator.CustomOpProp):
    def __init__(self, stride=0):
        super(RPNFPNOHEMProp, self).__init__(need_top_grad=False)
        self.stride = stride

    def list_arguments(self):
        return ['cls_score', 'bbox_weight', 'labels']

    def list_outputs(self):
        return ['labels_ohem', 'bbox_weights_ohem']

    def infer_shape(self, in_shape):
        labels_shape = in_shape[2]
        bbox_weights_shape = in_shape[1]
        #print('in_rpn_ohem', in_shape[0], in_shape[1], in_shape[2], file=sys.stderr)

        return in_shape, \
               [labels_shape, bbox_weights_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return RPNFPNOHEMOperator(self.stride)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
