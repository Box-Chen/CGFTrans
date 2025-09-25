# CGFTrans

This is the implementation of [CGFTrans: Cross-Modal Global Feature Fusion Transformer for Medical Report Generation](https://doi.org/10.1109/JBHI.2024.3414413) at JBHI-2024.

## Citation
If you find the code useful, please cite our paper:
~~~
@ARTICLE{10557585,
  author={Xu, Liming and Tang, Quan and Zheng, Bochuan and Lv, Jiancheng and Li, Weisheng and Zeng, Xianhua},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={CGFTrans: Cross-Modal Global Feature Fusion Transformer for Medical Report Generation}, 
  year={2024},
  volume={28},
  number={9},
  pages={5600-5612},
  keywords={Feature extraction;Transformers;Medical diagnostic imaging;Pathology;Visualization;Task analysis;Decoding;Medical report generation;transformer;multimodal learning;feature fusion;global feature},
  doi={10.1109/JBHI.2024.3414413}}
~~~

## Requirements

- `torch==1.7.1`
- `torchvision==0.8.2`
- `opencv-python==4.4.0.42`

## Datasets

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

For `MIMIC-CXR`, you can download the dataset from [here](https://physionet.org/content/mimic-cxr/2.0.0/) and then put the files in `data/mimic_cxr`.

## Train

Run `bash train_$dataset_name.sh` to train a model.

## Test

Run `bash test_$dataset_name.sh` to test a model.

## References

We thank [CMN](https://github.com/zhjohnchan/R2GenCMN) and [R2Gen](https://github.com/cuhksz-nlp/R2Gen) for their open source works.

