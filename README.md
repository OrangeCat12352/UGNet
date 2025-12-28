# Dual-Stream Feature Collaboration Perception Network for Salient Object Detection in Remote Sensing Images

⭐ This code has been completely released ⭐ 

⭐ our [article](https://www.mdpi.com/2079-9292/13/18/3755) ⭐ 

# 📖 Introduction
<span style="font-size: 125%">
As the core technology of artificial intelligence, salient object detection (SOD) is an important ap-proach to improve the analysis efficiency of remote sensing images by intelligently identifying key areas in images. However, existing methods that rely on a single strategy, convolution or Trans-former, exhibit certain limitations in complex remote sensing scenarios. Therefore, we developed a Dual-Stream Feature Collaboration Perception Network (DCPNet) to enable the collaborative work and feature complementation of Transformer and CNN. First, we adopted a dual-branch feature extractor with strong local bias and long-range dependence characteristics to perform multi-scale feature extraction from remote sensing images. Then, we presented a Multi-path Complemen-tary-aware Interaction Module (MCIM) to refine and fuse the feature representations of salient targets from the global and local branches, achieving fine-grained fusion and interactive alignment of dual-branch features. Finally, we proposed a Feature Weighting Balance Module (FWBM) to balance global and local features, preventing the model from overemphasizing global information at the expense of local details or from inadequately mining global cues due to excessive focus on local information. Extensive experiments on the EORSSD and ORSSD datasets demonstrated that DCPNet outperformed the current 19 state-of-the-art methods.
</span>
<p align="center"> <img src="Images/Figure 1.png" width=90%"></p>

If our code is helpful to you, please cite:

```
@Article{electronics13183755,
AUTHOR = {Li, Hongli and Chen, Xuhui and Mei, Liye and Yang, Wei},
TITLE = {Dual-Stream Feature Collaboration Perception Network for Salient Object Detection in Remote Sensing Images},
JOURNAL = {Electronics},
VOLUME = {13},
YEAR = {2024},
NUMBER = {18},
ARTICLE-NUMBER = {3755},
URL = {https://www.mdpi.com/2079-9292/13/18/3755},
ISSN = {2079-9292},
ABSTRACT = {As the core technology of artificial intelligence, salient object detection (SOD) is an important approach to improve the analysis efficiency of remote sensing images by intelligently identifying key areas in images. However, existing methods that rely on a single strategy, convolution or Transformer, exhibit certain limitations in complex remote sensing scenarios. Therefore, we developed a Dual-Stream Feature Collaboration Perception Network (DCPNet) to enable the collaborative work and feature complementation of Transformer and CNN. First, we adopted a dual-branch feature extractor with strong local bias and long-range dependence characteristics to perform multi-scale feature extraction from remote sensing images. Then, we presented a Multi-path Complementary-aware Interaction Module (MCIM) to refine and fuse the feature representations of salient targets from the global and local branches, achieving fine-grained fusion and interactive alignment of dual-branch features. Finally, we proposed a Feature Weighting Balance Module (FWBM) to balance global and local features, preventing the model from overemphasizing global information at the expense of local details or from inadequately mining global cues due to excessive focus on local information. Extensive experiments on the EORSSD and ORSSD datasets demonstrated that DCPNet outperformed the current 19 state-of-the-art methods.},
DOI = {10.3390/electronics13183755}
}
```
# Saliency maps
   We provide saliency maps of our and compared methods at [here](https://pan.baidu.com/s/1S3JdGOEv54g6e1IlXNqGdg?pwd=hmpg) on two datasets (ORSSD and EORSSD).
      
# DateSets
ORSSD download  at [here](https://github.com/rmcong/ORSSD-dataset)

EORSSD download at [here](https://github.com/rmcong/EORSSD-dataset)

The structure of the dataset is as follows:
```python
DCPNet
├── EORSSD
│   ├── train
│   │   ├── images
│   │   │   ├── 0001.jpg
│   │   │   ├── 0002.jpg
│   │   │   ├── .....
│   │   ├── lables
│   │   │   ├── 0001.png
│   │   │   ├── 0002.png
│   │   │   ├── .....
│   │   
│   ├── test
│   │   ├── images
│   │   │   ├── 0004.jpg
│   │   │   ├── 0005.jpg
│   │   │   ├── .....
│   │   ├── lables
│   │   │   ├── 0004.png
│   │   │   ├── 0005.png
│   │   │   ├── .....
```

# Train
1. Download the dataset.
2. Use data_aug.m to augment the training set of the dataset.

3. Download backbone weight at [pretrain](https://pan.baidu.com/s/1S3JdGOEv54g6e1IlXNqGdg?pwd=hmpg), and put it in './pretrain/'. 

4. Modify paths of datasets, then run train_MyNet.py.


# Test
1. Download the pre-trained models of our network at [weight](https://pan.baidu.com/s/1S3JdGOEv54g6e1IlXNqGdg?pwd=hmpg)
2. Modify paths of pre-trained models  and datasets.
3. Run test_MyNet.py.


# Visualization of results
<p align="center"> <img src="Images/Figure 4.png" width=95%"></p>

# Evaluation Tool
   You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.


# ORSI-SOD Summary
Salient Object Detection in Optical Remote Sensing Images Read List at [here](https://github.com/MathLee/ORSI-SOD_Summary)

# Acknowledgements
This code is built on [PyTorch](https://pytorch.org).
# Contact
If you have any questions, please submit an issue on GitHub or contact me by email (cxh1638843923@gmail.com).
       
                
