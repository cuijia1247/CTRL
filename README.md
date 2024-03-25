# StyleConsensusLearning

#### Description
Style Classification Community Feature Learning

#### Software Architecture
This project is used to classify artistic style by community feature learning

#### Installation

1.  python==3.9.18 
2.  pytorch==1.12.0 
3.  torchvision==0.13.0 
4.  pillow==10.0.1
5.  cudatoolkit==11.3.1
6.  cv2 # pip install opencv-python
7.  matploplib==3.8.2
8.  tqdm==4.66.1 # pip install tqdm
9.  matplatlib==3.8.2 # pip install matplotlib
10. scipy==1.11.4 # pip install scipy

#### Instructions

1.  create 'data' folder for training data, i.e., 'data/Painting91/'
2.  create 'features' folder for saving data features, i.e., 'features/painting91_vgg16.npy'
3.  create 'pretrainModels' for pretrained model downloaded form huggingface
All the files in 'data', 'features', 'pretrainedModels' and 'backup' are uploaded to Baidu YunPan.

#### Code Usages

The full codes will be released after the paper published. All the necessary files can be downloaded from the link below:
xxxxxxx
1. download 'Painting91' into 'data' folder;
2. download 'pretrainModel' folder into the project root;
3. download Painting91_resnet_CLS and Painting91_resnet_SLC files into the 'models' folder;
4. revise the configurations ini run_test.py
5. python run_test.py

#### Todo-list

- [x] finish the feature extraction code frame.
- [x] finish the dataloader for new dataset, such as painting91, pandora etc.
- [x] run the original dae project on painting91 for 1000 iterations
- [x] run the original dae codes on painting91 with smaller learning rate, 0.00001 ... , see the CLS-accuracy
- [x] test multi linear layers for classification model
- [x] add output log recording
- [x] delete the noise adding function on (1)
- [x] add the cdae module into 5, test the accuracy 
- [x] add more Conv2d layer for single cdae into 2, test the accuracy 
- [x] change the input into feature mode
- [x] test different loss function in sae model.py
- [x] how to identify the style level?
  - [x] try random set the style levels
  - [x] add level attribute in Dataset
  - [ ] <mark>the cluster algorithms maybe useful for potential key contribution</mark> potiental unsupervised style classification
- [x] add the style level computation into the painting91 to test the SSCAE
  - [x] make the generated new features into the circle
  - [x] add setLevels function in SCLdataset.py
  - [x] combine vgg16 and resnet50 sae class into one unified class
  - [x] combine image and features sae class into one unified class
  - [x] change the specified initialization process into the auto initialization with legal variables
  - [x] add ae2 ae3
  - [x] add different style level attribute in sae class
  - [x] find unsupervised cluster algorithm to set the style level
    - [x] compute the average centroid
    - [x] compute the distance from centroid
    - [x] finish the style level set based on Euclidean distance
  - [x] add different loss calculation for different sae class
  - [x] add the NN features calculation codes
  - [x] add classification loss into the sae loss, see what will happen
  - [ ] <mark>how to find the NN, maybe the potential key contribution</mark>
  - [x] re-write CDAutoEncoder class, the backward_pass should be changed
  - [x] find style similarity algorithms to replace the L2-norm
    - [x] customize loss class in 'utils/'
    - [x] try KL-diverge algorithm to calculate the style similarity
    - [x] try cos_similarity to calculate the style similarity
    - [x] try cityblock distance
    - [x] try chebyshev distance
    - [x] try DotProductSimilarity
  - [ ] try the diamond type of SLAE
    - [x] add diamond sae
    - [x] add dsc1, dsc2, dsc3
    - [x] try different activation function when the dsc layers are three
      - [x] try LeakReLU
      - [x] try ELU
      - [x] try SiLU
      - [x] try Mish
  - [x] try stacked one layer, instead the dsc1, dsc2, dsc3
  - [x] try two dsc1, see the CLS-accuracy
  - [x] re-write dsc1, l4->l3, l3->l2, l2->l1, l1->l1, re-do the upper two steps, see the CLS-accuracy
  - [x] try the parallel dsc model under the current circumstance
  - [x] try the jump-connection between different dsc cells
  - [x] try painting91 resnet50 2048x7x7, see the accuracy
  - [x] try pandora on vgg16 features
  - [x] try pandora on resnet50 features


  

