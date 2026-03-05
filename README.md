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

# Results
## Main results on ORSSD dataset






| **Methods** | **S<sub>α</sub>** |   **MAE**   | **adp E<sub>ξ</sub>** | **mean E<sub>ξ</sub>** | **max E<sub>ξ</sub>** | **adp F<sub>β</sub>** | **mean F<sub>β</sub>** | **max F<sub>β</sub>** |
|:-----------:|:-----------------:|:-----------:|:---------------------:|:----------------------:|:---------------------:|:---------------------:|------------------------|-----------------------|
|   SAMNet    |     	  0.8761     |   	0.0217   |        	0.8656        |        	0.8818         |        	0.9478        |        	0.6843        | 	0.7531                | 	0.8137               |
|   HVPNet    |     	  0.8610     |   	0.0225   |        	0.8471        |        	0.8717         |        	0.9320        |        	0.6726        | 	0.7396                | 	0.7938               |
|   DAFNet    |     	  0.9191     |   	0.0113   |        	0.9360        |        	0.9539         |        	0.9771        |        	0.7876        | 	0.8511                | 	0.8928               |
|   MSCNet    |     	  0.9227     |   	0.0129   |        	0.9584        |        	0.9653         |        	0.9754        |        	0.8350        | 	0.8676                | 	0.8927               |
|    MJRBM    |   	      0.9204   |   	0.0163   |        	0.9328        |        	0.9415         |        	0.9623        |        	0.8022        | 	0.8566                | 	0.8842               |
|    PAFR     |   	      0.8938   |   	0.0211   |        	0.9315        |        	0.9268         |        	0.9467        |        	0.8025        | 	0.8275                | 	0.8438               |
|   CorrNet   |     	  0.9380     |   	0.0098   |        	0.9721        |        	0.9746         |        	0.9790        |        	0.8875        | 	0.9002                | 	0.9129               |
|   EMFINet   |     	  0.9432     |   	0.0095   |        	0.9715        |        	0.9726         |        	0.9813        |        	0.8797        | 	0.9000                | 	0.9155               |
|   MCCNet    |     	  0.9437     |   	0.0087   |        	0.9735        |        	0.9758         |        	0.9800        |        	0.8957        | 	0.9054                | 	0.9155               |
|   ACCoNet   |     	  0.9437     |   	0.0088   |        	0.9721        |        	0.9754         |        	0.9796        |        	0.8806        | 	0.8971                | 	0.9149               |
|   AESINet   |     	  0.9460     |   	0.0086   |        	0.9707        |        	0.9747         |        	0.9828        |        	0.8666        | 	0.8986                | 	0.9183               |
|   ERPNet    |     	  0.9254     |   	0.0135   |        	0.9520        |        	0.8566         |        	0.9710        |        	0.8356        | 	0.8745                | 	0.8974               |
|   ADSTNet   |     	  0.9379     |   	0.0086   |        	0.9785        |        	0.9740         |        	0.9807        |        	0.8979        | 	0.9042                | 	0.9124               |
|   SFANet    |     	  0.9453     | 	**0.0070** |        	0.9765        |        	0.9789         |        	0.9830        |        	0.8984        | 	0.9063                | 	0.9192               |
|     VST     |   	      0.9365   |   	0.0094   |        	0.9466        |        	0.9621         |        	0.9810        |        	0.8262        | 	0.8817                | 	0.9095               |
|    ICON     |   	      0.9256   |   	0.0116   |        	0.9554        |        	0.9637         |        	0.9704        |        	0.8444        | 	0.8671                | 	0.8939               |
|   HFANet    |     	  0.9399     |   	0.0092   |        	0.9722        |        	0.9712         |        	0.9770        |        	0.8819        | 	0.8981                | 	0.9112               |
|  TLCKDNet   |      0.9421       |   	0.0082   |        	0.9696        |        	0.9710         |        	0.9794        |        	0.8719        | 	0.8947                | 	0.9114               |
|    ASNet    |   	      0.9441   |   	0.0081   |        	0.9795        |        	0.9764         |        	0.9803        |        	0.8986        | 	0.9072                | 	0.9172               |
|    Ours     | 	      **0.9498** |   	0.0073   |      	**0.9809**      |      	**0.9815**       |      	**0.9855**      |      	**0.9040**      | 	**0.9124**            | 	**0.9251**           |
- Bold indicates the best performance.

## Main results on EORSSD dataset

| **Methods** | **S<sub>α</sub>** |   **MAE**   | **adp E<sub>ξ</sub>** | **mean E<sub>ξ</sub>** | **max E<sub>ξ</sub>** | **adp F<sub>β</sub>** | **mean F<sub>β</sub>** | **max F<sub>β</sub>** |
|:-----------:|:-----------------:|:-----------:|:---------------------:|:----------------------:|:---------------------:|:---------------------:|------------------------|-----------------------|
|   SAMNet    |      	0.8622      |   	0.0132   |        	0.8284        |        	0.8700         |        	0.9421        |        	0.6114        | 	0.7214                | 	0.7813               |
|   HVPNet    |      	0.8734      |   	0.0110   |        	0.8270        |        	0.8721         |        	0.9482        |        	0.6202        | 	0.7377                | 	0.8036               |
|   DAFNet    |      	0.9166      |   	0.0060   |        	0.8443        |        	0.9290         |      	**0.9859**      |        	0.6423        | 	0.7842                | 	0.8612               |
|   MSCNet    |      	0.9071      |   	0.0090   |        	0.9329        |        	0.9551         |        	0.9689        |        	0.7553        | 	0.8151                | 	0.8539               |
|    MJRBM    |      	0.9197      |   	0.0099   |        	0.8897        |        	0.9350         |        	0.9646        |        	0.7066        | 	0.8239                | 	0.8656               |
|    PAFR     |      	0.8927      |   	0.0119   |        	0.8959        |        	0.9210         |        	0.9490        |        	0.7123        | 	0.7961                | 	0.8260               |
|   CorrNet   |      	0.9289      |   	0.0083   |        	0.9593        |        	0.9646         |        	0.9696        |        	0.8311        | 	0.8620                | 	0.8778               |
|   EMFINet   |      	0.9319      |   	0.0075   |        	0.9500        |        	0.9598         |        	0.9712        |        	0.8036        | 	0.8505                | 	0.8742               |
|   MCCNet    |      	0.9327      |   	0.0066   |        	0.9538        |        	0.9685         |        	0.9755        |        	0.8137        | 	0.8604                | 	0.8904               |
|   ACCoNet   |      0.9290       |   	0.0074   |        	0.9450        |        	0.9653         |        	0.9727        |        	0.7969        | 	0.8552                | 	0.8837               |
|   AESINet   |      	0.9358      |   	0.0079   |        	0.9462        |        	0.9636         |        	0.9751        |        	0.7923        | 	0.8524                | 	0.8838               |
|   ERPNet    |      	0.9210      |   	0.0089   |        	0.9228        |        	0.9401         |        	0.9603        |        	0.7554        | 	0.8304                | 	0.8632               |
|   ADSTNet   |      0.9311       |   	0.0065   |        	0.9681        |        	0.9709         |        	0.9769        |        	0.8532        | 	0.8716                | 	0.8804               |
|   SFANet    |      	0.9349      |   	0.0058   |        	0.9669        |        	0.9726         |        	0.9769        |        	0.8492        | 	0.8680                | 	0.8833               |
|     VST     |      	0.9208      |   	0.0067   |        	0.8941        |        	0.9442         |        	0.9743        |        	0.7089        | 	0.8263                | 	0.8716               |
|    ICON     |      	0.9185      |   	0.0073   |        	0.9497        |        	0.9619         |        	0.9687        |        	0.8065        | 	0.8371                | 	0.8622               |
|   HFANet    |      	0.9380      |   	0.0070   |        	0.9644        |        	0.9679         |        	0.9740        |        	0.8365        | 	0.8681                | 	0.8876               |
|  TLCKDNet   |      0.9350       |   	0.0056   |        	0.9514        |        	0.9661         |        	0.9788        |        	0.7969        | 	0.8535                | 	0.8843               |
|    ASNet    |      	0.9345      |   	0.0055   |        	0.9748        |        	0.9745         |        	0.9783        |        	0.8672        | 	0.8770                | 	**0.8959**           |
|    Ours     |    	**0.9408**    | 	**0.0053** |      	**0.9772**      |      	**0.9773**       |        	0.9817        |      	**0.8695**      | 	**0.8812**            | 	0.8936               |
- Bold indicates the best performance.


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
       
                
