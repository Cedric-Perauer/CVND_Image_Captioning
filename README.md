# CVND---Image-Captioning-Project

# Instructions  
1. Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

2. Setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```

3. Download some specific data from here: http://cocodataset.org/#download (described below)

* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)

4. The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order (`0_Dataset.ipynb, 1_Preliminaries.ipynb, 2_Training.ipynb, 3_Inference.ipynb`).

## Implementation 
- based on the paper [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555.pdf)
- propose an LSTM based sentence Generator that acts as a Decoder for a pre trained CNN (in this case ResNet 50)
- last hidden layer serves as RNN Input, the feature vector is flattened in order to serve as the LSTM Input 
- the Decoder as well as the Encoder Network Output Layer were trained/retrained

## Proposed Improvements of the implementation  
- Beam Search 
- different Encoder Network (Faster RCNN, YOLO, RetinaNet,...)
- experiment with different Encoder Weights (Image Net instead of COCO)
- add Dropout as proposed by the paper
- Different RNN Architecture (GRU, tweak LSTM Layer)
- Use of Attention Mechanism [Luong](https://towardsdatascience.com/sequence-2-sequence-model-with-attention-mechanism-9e9ca2a613a)
- Replace RNN with a [Transformer architecture](https://papers.nips.cc/paper/9293-image-captioning-transforming-objects-into-words.pdf)
