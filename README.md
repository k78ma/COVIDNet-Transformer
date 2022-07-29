# COVIDNet-Transformer

Full paper: https://arxiv.org/abs/2207.09312

## Abstract
Building AI models with trustworthiness is important especially in regulated areas such as healthcare. In tackling COVID-19, previous work uses convolutional neural networks as the backbone architecture, which has shown to be prone to over-caution and overconfidence in making decisions, rendering them less trustworthy -- a crucial flaw in the context of medical imaging. In this study, we propose a feature learning approach using Vision Transformers, which use an attention-based mechanism, and examine the representation learning capability of Transformers as a new backbone architecture for medical imaging. Through the task of classifying COVID-19 chest radiographs, we investigate into whether generalization capabilities benefit solely from Vision Transformers' architectural advances. Quantitative and qualitative evaluations are conducted on the trustworthiness of the models, through the use of "trust score" computation and a visual explainability technique. We conclude that the attention-based feature learning approach is promising in building trustworthy deep learning models for healthcare.

## Results

*Table 1. Precision scores on the unseen COVIDx V9B test split.
The best results in each class are bolded.*
| Model                 | Negative  | Positive  |
|-----------------------|-----------|-----------|
| ResNet (200 epochs)   | **0.952** | **1.000** |
| DenseNet (200 epochs) | 0.948     | 0.995     |
| Swin-B (30 epochs)    | 0.926     | **1.000** |
| Swin-B (50 epochs)    | 0.935     | **1.000** |
| Swin-B (100 epochs)   | 0.930     | **1.000** |
| Swin-B (200 epochs)   | **0.952** | **1.000** |

<br>

*Table 2. Sensitivity scores on the unseen COVIDx V9B test split.
The best results in each class are bolded.*
| Model                 | Negative  | Positive  |
|-----------------------|-----------|-----------|
| ResNet (200 epochs)   | **1.000** | **0.950** |
| DenseNet (200 epochs) | 0.995     | 0.945     |
| Swin-B (30 epochs)    | **1.000** | 0.920     |
| Swin-B (50 epochs)    | **1.000** | 0.930     |
| Swin-B (100 epochs)   | **1.000** | 0.925     |
| Swin-B (200 epochs)   | **1.000** | **0.950** |

<br>

*Table 3. Trust scores calculated from each experiment on the positive class. The best result is bolded.*
| Model                 | Trust Score |
|-----------------------|-------------|
| ResNet (200 epochs)   | 0.923       |
| DenseNet (200 epochs) | 0.922       |
| Swin-B (30 epochs)    | 0.943       |
| Swin-B (50 epochs)    | 0.959       |
| Swin-B (100 epochs)   | 0.954       |
| Swin-B (200 epochs)   | **0.963**   |

<br>

_Figure 1. Swin-B and ResNet-50 Ablation-CAMs for 3 selected COVID-positive chest X-rays. ResNet-50 is chosen as a representative because it produced better results and localization maps than Densenet-121. Warm colors (red, orange) indicate high importance, cold colors (blue, green) indicate lower importance._

![image](https://user-images.githubusercontent.com/77073162/181854254-f802c1f5-7807-45be-8947-38d24714e924.png)

# Training Instructions

# Other

### Contact
- <k78ma@uwaterloo.ca>
- <pengcheng.xi_at_nrc-cnrc.gc.ca>

### Citation
```
@misc{https://doi.org/10.48550/arxiv.2207.09312,
  doi = {10.48550/ARXIV.2207.09312},
  
  url = {https://arxiv.org/abs/2207.09312},
  
  author = {Ma, Kai and Xi, Pengcheng and Habashy, Karim and Ebadi, Ashkan and Tremblay, Stéphane and Wong, Alexander},
  
  keywords = {Image and Video Processing (eess.IV), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Towards Trustworthy Healthcare AI: Attention-Based Feature Learning for COVID-19 Screening With Chest Radiography},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```
