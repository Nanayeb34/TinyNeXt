# TinyNeXt

---
Official pytorch implementation of "**[An Efficient Hybrid Vision Transformer for TinyML Applications, ICCV'2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zeng_An_Efficient_Hybrid_Vision_Transformer_for_TinyML_Applications_ICCV_2025_paper.pdf)**"

> **Abstract:** To enable the deployment of Vision Transformers on resource-constrained mobile and edge devices, the development of efficient ViT models has attracted significant attention. Researchers achieved remarkable improvements in accuracy and speed by optimizing attention mechanisms and integrating lightweight CNN modules. However, existing designs often overlook runtime overhead from memory-bound operations and the shift in feature characteristics from spatial-dominant to semantic-dominant as networks deepen. This work introduces TinyNeXt, a family of efficient hybrid ViTs for TinyML, featuring Lean Single-Head Self-Attention to minimize memory-bound operations, and a macro design tailored to feature characteristics at different stages. TinyNeXt strikes a better accuracy-speed trade-off across diverse tasks and hardware platforms, outperforming state-of-the-art models of comparable scale. For instance, our TinyNeXt-T achieves a remarkable 71.5% top-1 accuracy with only 1.0M parameters on ImageNet-1K. Furthermore, compared to recent efficient models like MobileViT-XXS and MobileViT-XS, TinyNeXt-S and TinyNeXt-M achieve 3.7%/0.5% higher accuracy, respectively, while running 2.1×/2.6× faster on Nvidia Jetson Nano.

------
<p align="center">
  <img src="fig\top1.png" width=80%> <br>
  Comparison with SOTA models on ImgeNet-1K
</p>
<p align="center">
  <img src="fig\arch.png" width=90%> <br>
  Overview of TinyNeXt
</p>

------

## Repository Structure

- [classification](classification/README.md)/: Image classification related code and checkpoint.
- [detection](detection/README.md)/: Object detection related code.
- [segmentation](segmentation/README.md)/: Semantic segmentation related code .
- [speed_benchmark](speed_benchmark/README.md)/: Throughput testing scripts and logs

## Model Performance

### Image Classification Performance (ImageNet-1K)

| Model | Top-1 Accuracy | Parameters | MACs | Latency |
|:--------------:|:----:|:---:|:--:|:--:|
| TinyNeXt-M | 75.3% | 2.3M | 475M | 19.4ms |
| TinyNeXt-S | 72.7% | 1.3M | 304M | 14.3ms |
| TinyNeXt-T | 71.5% | 1.0M | 259M | 12.7ms |

Latency is measured on Nvidia Jetson Nano.

### Object Detection Performance Based on SSDLite (MS-COCO 2017)

| Backbone | AP | AP<sub>50</sub> | AP<sub>75</sub> | Parameters |
| :--------: | :--: | :-------------: | :-------------: | :-----: |
| TinyNeXt-S | 22.4 | 37.9 | 22.7 | 2.3M |
| TinyNeXt-M | 25.0 | 41.1 | 25.4 | 3.3M |

### Semantic Segmentation Performance Based on DeepLabv3 (Pascal VOC 2012)

|  Backbone  | Parameters | Flops | mIOU |
| :--------: | :--------: | :---: | :--: |
| TinyNeXt-S |    2.3M    | 3.5G  | 75.5 |
| TinyNeXt-M |    3.3M    | 5.1G  | 76.9 |

## Acknowledgements
We thank but not limited to following repositories for providing assistance for our research:
- [PyTorch](https://pytorch.org/)
- [TIMM](https://github.com/rwightman/pytorch-image-models)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)

## Citation

``````
@InProceedings{Zeng_2025_ICCV,
	author    = {Zeng, Fanhong and Li, Huanan and Guan, Juntao and Fan, Rui and Wu, Tong and Wang, Xilong and Lai, Rui},
	title     = {An Efficient Hybrid Vision Transformer for TinyML Applications},
	booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
	month     = {October}, 
	year      = {2025},    
	pages     = {19914-19924} 
}
``````

