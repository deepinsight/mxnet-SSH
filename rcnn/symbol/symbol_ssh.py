import mxnet as mx
import numpy as np
from rcnn.config import config
from rcnn.PY_OP import fpn_roi_pooling, proposal_fpn, mask_roi, mask_output, proposal_fpn_out, rpn_fpn_ohem
FPN = False
USE_DCN=False

def conv_act_layer(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type="relu", bias_wd_mult=0.0, dcn=False):

    weight = mx.symbol.Variable(name="{}_weight".format(name),   
        init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
    bias = mx.symbol.Variable(name="{}_bias".format(name),   
        init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0', '__wd_mult__': str(bias_wd_mult)})
    if not dcn:
      conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
          stride=stride, num_filter=num_filter, name="{}".format(name), weight = weight, bias=bias)
    else:
      assert kernel[0]==3 and kernel[1]==3
      num_group = 1
      f = num_group*18
      offset_weight = mx.symbol.Variable(name="{}_offset_weight".format(name),   
          init=mx.init.Constant(0.0), attr={'__lr_mult__': '1.0'})
      offset_bias = mx.symbol.Variable(name="{}_offset_bias".format(name),   
          init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0', '__wd_mult__': str(bias_wd_mult)})
      conv_offset = mx.symbol.Convolution(name=name+'_offset', data = from_layer, weight=offset_weight, bias=offset_bias,
                          num_filter=f, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
      conv = mx.contrib.symbol.DeformableConvolution(name=name, data=from_layer, offset=conv_offset, weight=weight, bias=bias,
                          num_filter=num_filter, pad=(1,1), kernel=(3, 3), num_deformable_group=num_group, stride=(1, 1), no_bias=False)
    if len(act_type)>0:
      relu = mx.symbol.Activation(data=conv, act_type=act_type, \
          name="{}_{}".format(name, act_type))
    else:
      relu = conv
    return relu

def ssh_context_module(body, num_filters, name):
  conv_dimred = conv_act_layer(body, name+'_conv1',
      num_filters, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', dcn=False)
  conv5x5 = conv_act_layer(conv_dimred, name+'_conv2',
      num_filters, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='', dcn=USE_DCN)
  conv7x7_1 = conv_act_layer(conv_dimred, name+'_conv3_1',
      num_filters, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', dcn=False)
  conv7x7 = conv_act_layer(conv7x7_1, name+'_conv3_2',
      num_filters, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='', dcn=USE_DCN)
  return (conv5x5, conv7x7)

def ssh_detection_module(body, num_filters, name):
  conv3x3 = conv_act_layer(body, name+'_conv1',
      num_filters, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='', dcn=USE_DCN)
  conv5x5, conv7x7 = ssh_context_module(body, num_filters//2, name+'_context')
  ret = mx.sym.concat(*[conv3x3, conv5x5, conv7x7], dim=1, name = name+'_concat')
  ret = mx.symbol.Activation(data=ret, act_type='relu', name=name+'_concat_relu')
  return ret

def get_feat_down(conv_feat):
    #P5 = mx.symbol.Convolution(data=conv_feat[0], kernel=(1, 1), num_filter=256, name="P5_lateral")
    P5 = conv_act_layer(conv_feat[0], 'P5_lateral',
        256, kernel=(1,1), pad=(0,0), stride=(1, 1), act_type='relu')

    # P5 2x upsampling + C4 = P4
    P5_up   = mx.symbol.UpSampling(P5, scale=2, sample_type='nearest', workspace=512, name='P5_upsampling', num_args=1)
    #P4_la   = mx.symbol.Convolution(data=conv_feat[1], kernel=(1, 1), num_filter=256, name="P4_lateral")
    P4_la = conv_act_layer(conv_feat[1], 'P4_lateral',
        256, kernel=(1,1), pad=(0,0), stride=(1, 1), act_type='relu')
    P5_clip = mx.symbol.Crop(*[P5_up, P4_la], name="P4_clip")
    P4      = mx.sym.ElementWiseSum(*[P5_clip, P4_la], name="P4_sum")
    #P4      = mx.symbol.Convolution(data=P4, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P4_aggregate")
    P4 = conv_act_layer(P4, 'P4_aggregate',
        256, kernel=(3,3), pad=(1,1), stride=(1, 1), act_type='relu')

    # P4 2x upsampling + C3 = P3
    P4_up   = mx.symbol.UpSampling(P4, scale=2, sample_type='nearest', workspace=512, name='P4_upsampling', num_args=1)
    #P3_la   = mx.symbol.Convolution(data=conv_feat[2], kernel=(1, 1), num_filter=256, name="P3_lateral")
    P3_la = conv_act_layer(conv_feat[2], 'P3_lateral',
        256, kernel=(1,1), pad=(0,0), stride=(1, 1), act_type='relu')
    P4_clip = mx.symbol.Crop(*[P4_up, P3_la], name="P3_clip")
    P3      = mx.sym.ElementWiseSum(*[P4_clip, P3_la], name="P3_sum")
    #P3      = mx.symbol.Convolution(data=P3, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P3_aggregate")
    P3 = conv_act_layer(P3, 'P3_aggregate',
        256, kernel=(3,3), pad=(1,1), stride=(1, 1), act_type='relu')

    return P3, P4, P5

def get_ssh_conv(data):
    """
    shared convolutional layers
    :param data: Symbol
    :return: Symbol
    """
    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv5_3")
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    m3_pool = mx.sym.Pooling(data=relu5_3, kernel=(2, 2), stride=(2,2), pad=(0,0), pool_type='max')
    if FPN:
      relu4_3, relu5_3, m3_pool = get_feat_down([m3_pool, relu5_3, relu4_3])

    F1 = 256
    F2 = 128
    _bwm = 1.0
    conv4_128 = conv_act_layer(relu4_3, 'ssh_m1_red_conv',
        F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
    conv5_128 = conv_act_layer(relu5_3, 'ssh_m2_red_conv',
        F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
    conv5_128_up = mx.symbol.Deconvolution(data=conv5_128, num_filter=F2, kernel=(4,4),  stride=(2, 2), pad=(1,1),
        num_group = F2, no_bias = True, attr={'__lr_mult__': '0.0', '__wd_mult__': '0.0'},
        name='ssh_m2_red_upsampling')
    #conv5_128_up = mx.symbol.Deconvolution(data=conv5_128, num_filter=F2, kernel=(2,2),  stride=(2, 2), pad=(0,0),
    #    num_group = F2, no_bias = True, attr={'__lr_mult__': '0.0', '__wd_mult__': '0.0'},
    #    name='ssh_m2_red_upsampling')
    conv4_128 = mx.symbol.Crop(*[conv4_128, conv5_128_up])
    #conv5_128_up = mx.symbol.Crop(*[conv5_128_up, conv4_128])

    conv_sum = conv4_128+conv5_128_up
    #conv_sum = conv_1x1

    m1_conv = conv_act_layer(conv_sum, 'ssh_m1_conv',
        F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
    m1 = ssh_detection_module(m1_conv, F2, 'ssh_m1_det')
    m2 = ssh_detection_module(relu5_3, F1, 'ssh_m2_det')
    m3 = ssh_detection_module(m3_pool, F1, 'ssh_m3_det')
    return {8: m1, 16:m2, 32: m3}


def get_ssh_train():
    """
    Region Proposal Network with VGG
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    var_label = False
    var_bbox_weight = False

    # shared convolutional layers
    conv_fpn_feat = get_ssh_conv(data)
    rpn_cls_score_list = []
    rpn_bbox_pred_list = []
    ret_group = []
    for stride in config.RPN_FEAT_STRIDE:
      num_anchors = config.RPN_ANCHOR_CFG[str(stride)]['NUM_ANCHORS']
      label = mx.symbol.Variable(name='label_stride%d'%stride)
      bbox_target = mx.symbol.Variable(name='bbox_target_stride%d'%stride)
      bbox_weight = mx.symbol.Variable(name='bbox_weight_stride%d'%stride)
      rpn_relu = conv_fpn_feat[stride]
      if not config.USE_MAXOUT or stride!=config.RPN_FEAT_STRIDE[-1]:
        rpn_cls_score = conv_act_layer(rpn_relu, 'rpn_cls_score_stride%d'%stride, 2*num_anchors,
            kernel=(1,1), pad=(0,0), stride=(1, 1), act_type='')
      else:
        cls_list = []
        for a in range(num_anchors):
          rpn_cls_score_bg = conv_act_layer(rpn_relu, 'rpn_cls_score_stride%d_anchor%d_bg'%(stride,a), 3,
              kernel=(1,1), pad=(0,0), stride=(1, 1), act_type='')
          rpn_cls_score_bg = mx.sym.max(rpn_cls_score_bg, axis=1, keepdims=True)
          cls_list.append(rpn_cls_score_bg)
          rpn_cls_score_fg = conv_act_layer(rpn_relu, 'rpn_cls_score_stride%d_anchor%d_fg'%(stride,a), 1,
              kernel=(1,1), pad=(0,0), stride=(1, 1), act_type='')
          cls_list.append(rpn_cls_score_fg)
        rpn_cls_score = mx.sym.concat(*cls_list, dim=1)
      rpn_bbox_pred = conv_act_layer(rpn_relu, 'rpn_bbox_pred_stride%d'%stride, 4*num_anchors,
          kernel=(1,1), pad=(0,0), stride=(1, 1), act_type='')

      # prepare rpn data
      rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                                shape=(0, 2, -1),
                                                name="rpn_cls_score_reshape_stride%s" % stride)
      rpn_bbox_pred_reshape = mx.symbol.Reshape(data=rpn_bbox_pred,
                                                shape=(0, 0, -1),
                                                name="rpn_bbox_pred_reshape_stride%s" % stride)

      if config.TRAIN.RPN_ENABLE_OHEM<0:
        rpn_bbox_pred_list.append(rpn_bbox_pred_reshape)
        rpn_cls_score_list.append(rpn_cls_score_reshape)
      else:
        if config.TRAIN.RPN_ENABLE_OHEM==2:
          label, bbox_weight = mx.sym.Custom(op_type='rpn_fpn_ohem', stride=int(stride), cls_score=rpn_cls_score_reshape, bbox_weight = bbox_weight , labels = label)
        #label_list.append(label)
        #bbox_weight_list.append(bbox_weight)
        rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape,
                                               label=label,
                                               multi_output=True,
                                               normalization='valid', use_ignore=True, ignore_label=-1,
                                               name='rpn_cls_prob_stride%d'%stride)

        bbox_diff = rpn_bbox_pred_reshape-bbox_target
        bbox_diff = bbox_diff * bbox_weight
        rpn_bbox_loss_ = mx.symbol.smooth_l1(name='rpn_bbox_loss_stride%d_'%stride, scalar=3.0, data=bbox_diff)

        rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss_stride%d'%stride, data=rpn_bbox_loss_, grad_scale=1.0 / (config.TRAIN.RPN_BATCH_SIZE))
        ret_group.append(rpn_cls_prob)
        ret_group.append(mx.sym.BlockGrad(label))
        ret_group.append(rpn_bbox_loss)
        ret_group.append(mx.sym.BlockGrad(bbox_weight))

    return mx.sym.Group(ret_group)


