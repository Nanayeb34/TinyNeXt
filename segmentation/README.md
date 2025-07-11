# Semantic Segmentation Implementation
## Semantic Segmentation Performance Based on [DeepLabv3](https://arxiv.org/pdf/1706.05587.pdf) for [Pascal VOC 2012 ](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/):
| Backbone   | #Params | Flops | mIOU |                             Log                              |                             ckpt                             |
| ---------- | :-----: | :---: | :--: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| TinyNeXt-S |  2.3M   | 3.5G  | 75.5 | [log](work_dirs/deeplabv3_tinynext_s-40k_voc-aug-512x512/20250516_144652/20250516_144652.log) | [ckpt](https://huggingface.co/yuffeenn/TinyNeXt/blob/main/deeplabv3_tinynext_s-40k_voc-aug-512x512.pth) |
| TinyNeXt-M |  3.3M   | 5.1G  | 76.9 | [log](work_dirs/deeplabv3_tinynext_m-40k_voc-aug-512x512/20250516_160233/20250516_160233.log) | [ckpt](https://huggingface.co/yuffeenn/TinyNeXt/blob/main/deeplabv3_tinynext_m-40k_voc-aug-512x512.pth) |



## Get Start

1.Ref to [MMsegmentation](https://github.com/open-mmlab/mmsegmentation) for the environments.

2.Configs can be found in `./configs/ssd_tinynext`

3.Backbone checkpoint can be found in `../classification/logs`

4.run :

```
./dish_train.sh configs/deeplabv3/deeplabv3_tinynext_m-40k_voc-aug-512x512.py 4
```

or

```
./dish_train.sh configs/deeplabv3/deeplabv3_tinynext_t-40k_voc-aug-512x512.py 4
```

