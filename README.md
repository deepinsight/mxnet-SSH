# mxnet-SSH
Reproduce [SSH](https://arxiv.org/abs/1708.03979) (Single Stage Headless Face Detector) with MXNet.

Original Caffe code: [https://github.com/mahyarnajibi/SSH](https://github.com/mahyarnajibi/SSH)

Evaluation on WIDER FACE:

| Impelmentation     | Easy-Set | Medium-Set | Hard-Set |
| ------------------ | -------- | ---------- | -------- |
| Original Caffe SSH | 0.93123  | 0.92106    | 0.84582  |
| Our SSH Model      | 0.93489  | 0.92281    | 0.84525  |

### Installation
1. Clone the repository.

2. Download MXNet VGG16 pretrained model from [here](http://data.dmlc.ml/models/imagenet/vgg/vgg16-0000.params) and put it under `model` directory.

3. Type `make` to build necessary cxx libs.

### Training

```
python train_ssh.py
```

For more information, check our insightface repository.
