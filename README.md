Probabilistic Uncertainty-Guided Salient Object Detection in Remote Sensing Images

⭐ This code has been completely released ⭐ 

⭐ our [article](https://www.ingentaconnect.com/contentone/asprs/pers/2026/00000092/00000002/art00015) ⭐ 

# 📖 Introduction
<span style="font-size: 125%">
Salient object detection holds significant application value in fields such as agricultural monitoring, disaster assessment, and urban planning, providing critical support for precise decision-making. Existing deep learning‐based detection methods often rely on nonlinear mappings to perform binary classification of pixels. However, considerable uncertainty often occurs in areas where objects and backgrounds are alike because of lighting variations, shadow effects, and similar object interference. This uncertainty negatively affects the detection performance, especially at pixels near the decision boundary. To address this issue, a remote sensing salient object-detection method is proposed based on probabilistic uncertainty assessment (uncertainty guided network [UGNet]). First, a multi-scale encoder‐decoder framework with deep supervision is designed for the uncertainty calculation of confusing features. It uses high-level semantic features as guidance to enhance the ability to distinguish confusing features. Then, an uncertainty estimation mapping module is constructed, which uses Gaussian distribution to weight uncertain pixels, thereby improving the semantic distinction in confusing regions. A multi-scale focus fusion module is then introduced to integrate global and local information, reducing the uncertainty of multi-scale confusing features. Finally, multi-scale deep supervision is used to enhance the accuracy of salient object detection. Experimental results on two public data sets, optical remote sensing saliency detection and extended optical remote sensing saliency detection, demonstrate that the proposed UGNet outperforms 18 mainstream methods, with significantly improved detection performance.
</span>
<p align="center"> <img src="Images/Figure 1.png" width=90%"></p>

# DateSets
ORSSD download  at [here](https://github.com/rmcong/ORSSD-dataset)

EORSSD download at [here](https://github.com/rmcong/EORSSD-dataset)

The structure of the dataset is as follows:
```python
UGNet
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

3. Download backbone weight at [pvt_v2_b2.pth](https://pan.baidu.com/s/16YBJFEUu7JB0lE_7fpPFSg?pwd=g4vd), and put it in './pretrain/'. 

4. Modify paths of datasets, then run train_MyNet.py.


# Test
1. Download the pre-trained models of our network at [weight](https://pan.baidu.com/s/16YBJFEUu7JB0lE_7fpPFSg?pwd=g4vd)
2. Modify paths of pre-trained models  and datasets.
3. Run test_MyNet.py.


# Evaluation Tool
   You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.


# ORSI-SOD Summary
Salient Object Detection in Optical Remote Sensing Images Read List at [here](https://github.com/MathLee/ORSI-SOD_Summary)

# Acknowledgements
This code is built on [PyTorch](https://pytorch.org).
# Contact
If you have any questions, please submit an issue on GitHub or contact me by email (cxh1638843923@gmail.com).
       
                
