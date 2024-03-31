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

The full codes will be released after the paper published. All the necessary files can be downloaded from the Baidu Yun link below:
Link：https://pan.baidu.com/s/19rkaIOJNgKH3VY2y0nGlKg 
Code：dwsg
1. download 'Painting91' into 'data' folder;
2. download 'pretrainModel' folder into the project root;
3. download Painting91_resnet_CLS and Painting91_resnet_SLC files into the 'models' folder;
4. revise the configurations ini run_test.py
5. python run_test.py
The training codes and pre-trained models will be released soon.

#### Experiment Results with SOTA

|                     | Painting91 | Pandora | WikiArt3 | Arch  | FashionStyle14 | AVAStyle |
|---------------------|------------|---------|----------|-------|----------------|----------|
| VGG16               | 58.42      | 49.73   | 40.02    | 61.41 | 68.22          | 39.94    |
| VGG19               | 58.11      | 46.44   | 39.93    | 60.11 | 66.14          | 40.02    |
| ResNet50            | 64.93      | 51.65   | 47.01    | 65.12 | 71.13          | 40.05    |
| Resnet101           | 65.50      | 52.61   | 46.11    | 66.42 | 70.00          | 47.02    |
| InceptionV3         | 53.41      | 42.83   | 36.68    | 61.52 | 62.70          | 33.33    |
| DAE                 | 58.82      | 48.71   | 41.48    | 58.55 | 61.48          | 41.46    |
| SAE                 | 63.65      | 48.64   | 41.53    | 59.61 | 74.33          | 40.29    |
| SSCAE               | 64.07      | 49.38   | 43.65    | 60.48 | **75.02**          | 45.77    |
| DDS                 | 62.21      | 52.35   | 43.17    | /     | /              | /        |
| MCFFNet             | 66.60      | 51.39   | 45.51    | **66.12** | 68.38          | 42.69    |
| STACLE              | 60.41      | 55.80   | 47.21    | 60.81 | 64.47          | 46.38    |
| TCFL+VGG16(TOP1)    | **67.39**      | **56.67**   | **47.85**    | 65.57 | 71.67          | **47.22**    |
| TCFL+VGG16(TOP2)    | 83.19      | 74.92   | 66.99    | 78.12 | 85.69          | 63.92    |
| TCFL+VGG16(TOP3)    | 92.27      | 84.52   | 78.22    | 85.05 | 91.11          | 75.74    |
| TCFL+ResNet50(TOP1) | **69.12**      | **56.98**   | **51.62**    | **69.03** | **77.17**          | **53.94**    |
| TCFL+ResNet50(TOP2) | 85.29      | 76.15   | 69.11    | 81.82 | 87.44          | 68.68    |
| TCFL+ResNet50(TOP3) | 92.27      | 84.52   | 78.22    | 85.05 | 91.10          | 75.74    |


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
  - [x] re-write CDAutoEncoder class, the backward_pass should be changed
  - [x] find style similarity algorithms to replace the L2-norm
    - [x] customize loss class in 'utils/'
    - [x] try KL-diverge algorithm to calculate the style similarity
    - [x] try cos_similarity to calculate the style similarity
    - [x] try cityblock distance
    - [x] try chebyshev distance
    - [x] try DotProductSimilarity
  - [x] try the diamond type of SLAE
    - [x] add diamond sae
    - [x] add dsc1, dsc2, dsc3
    - [x] try different activation function when the dsc layers are three
      - [x] try LeakReLU
      - [x] try ELU
      - [x] try SiLU
      - [x] try Mish
  - [x] try stacked one layer, instead the dsc1, dsc2, dsc3
  - [x] try two dsc1, see the CLS-accuracy
  - [x] try the parallel dsc model under the current circumstance
  - [x] try the jump-connection between different dsc cells

#### Contributor




  

