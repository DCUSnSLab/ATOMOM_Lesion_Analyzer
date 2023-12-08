# Skin Lesion Detection System
![image](https://github.com/DCUSnSLab/ATOMOM_Lesion_Analyzer/assets/48535768/983e5510-2e93-413b-bf83-3bb55433c607)

# Ensemble-based Skin Lesion Detection System

## Overview
This repository contains the code and resources for an Ensemble-based Skin Lesion Detection System designed to assist in the identification and classification of skin lesions through image analysis. This system leverages advanced machine learning techniques to provide accurate and timely diagnostics.

If you use our code, please use it after installation through the .yaml file we uploaded
## System Architecture
The system is composed of the following key modules:

### Data Exchange Module
- Image Receiver: Receives images for analysis.
- Diagnostic Result Sender: Sends the diagnosis results to the user.

### Diagnostic Result Analysis Module
- Grad-CAM Visualization: Provides a heatmap visualization for identifying the focus areas of the model.
- Lesion Segmentation: Segments the lesion from the skin image.

### Skin Lesion Inference Module
- Lesion Detection: Detects the presence of a lesion in the skin image.
- Lesion Classification: Classifies the type of lesion (e.g., atopic dermatitis, insect bite, psoriasis).
- Ensemble Decision: Uses soft voting to combine the results of multiple models to improve accuracy.

## Getting Started
### Dependency
- This work was tested with PyTorch 1.10.1,Tensorflow 2.2.0 CUDA 11.2, python 3.7.13, django 3.2.22, djangorestframework 3.14, django-extensions 3.2.3 and Ubuntu 18.04.3 LTS. 

```
You must need `conda env create --file environment.yaml`.
```

### Demo instruction using pre-trained model
- Currently, this repository does not provide a pre-trained model for public use due to size constraints and licensing issues. If you are interested in a demonstration of the Ensemble-based Skin Lesion Detection System using a pre-trained model, please follow these steps:
 ```
1. Obtain a pre-trained model by contacting the repository maintainers at vhrxksro@gmail.com or marsberry@cu.ac.kr.
2. Place the pre-trained model in the designated directory within the project.
 ```
* Run with pretrained model
``` (with python 3.7)
1. Download pretrained models 
2. Move pre-trained efficientnet_models model into `classification/efficientnet_models`
3. Move pre-trained Mask R-CNN and YOLOv8 into `segmentation/mrcnn_models and segmentation/yolo_models`
5. python server/manage.py runserver [your ip:port]
```
### Dataset Information and Usage

The datasets used in this Ensemble-based Skin Lesion Detection System can be accessed through the following AI Hub URLs:

- [AI Hub Dataset 1(피부 질환 진단 의료 이미지](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=230)
- [AI Hub Dataset 2(소아청소년 피부질환 이미지 데이터)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=508)

To access and use these datasets, please ensure that you follow the rules and regulations provided by AI Hub. This may include procedures for registration, data handling, privacy compliance, and licensing agreements.

Note: The datasets provided by AI Hub are subject to their terms of use, and it is the responsibility of the users to adhere to these terms. If you have any questions about the usage of the datasets, please refer to the AI Hub's guidelines or contact their support for assistance.





### Sample Results

The images below demonstrate the sample results from the ATOMOM Lesion Analyzer for both desktop and mobile interfaces.

#### Desktop Interface Result
![image](https://github.com/DCUSnSLab/ATOMOM_Lesion_Analyzer/assets/48535768/e06ba319-df2c-4d7d-90c3-6bf73a29f10c)


#### Mobile Interface Result
![image](https://github.com/DCUSnSLab/ATOMOM_Lesion_Analyzer/assets/48535768/adb62cfe-f2ba-4bca-88ae-a54540d21540)




Note: If the images are not displaying, please ensure that the links are correct and that the GitHub repository settings allow for image hosting.

## Acknowledgements
This implementation has been based on these repository [yolov8](https://github.com/ultralytics/ultralytics), [
Mask R-CNN](https://github.com/leekunhee/Mask_RCNN), [EfficientNet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch).


