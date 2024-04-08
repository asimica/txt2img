## Image generation based on image description using artificial intelligence <br>
This repository shows work that investigates the creation of a system for image generation based on a given image description. <br>
For this purpose, several different artificial intelligence models that generate the most accurate and different image for a given description are designed, implemented and trained. <br>
Architectures based on GAN networks and transformers are examined.  <br>
It will use existing datasets such as COCO, CUB, ImageNet and similar to train the model.  <br>
The models will be evaluated with appropriate metrics in order to select the most efficient model. <br>
<br>
Currently, the repository consists of several python files that implement the training process of GAN networks. The file <code>gan64.py</code> represents the training process of a GAN network capable of generating images of size 64x64. 
Both the generator and disriminator are neural networks consisted of 5 convolutional layers. The network was trained on two datasets, the [flickr30k](https://paperswithcode.com/paper/from-image-descriptions-to-visual-denotations) dataset, and the [CUB 200](https://paperswithcode.com/dataset/cub-200-2011) dataset. 
The file <code>gan128.py</code> is the training process of a GAN network that outputs images of size 128x128. The architecture of this network is consisted of a generator and discriminator which are neural networks with 6 convolutional layers. 
Some of the images generated in the training process of this network are shown below. 

![Screenshot_1](https://github.com/asimica/txt2img/assets/97536578/3314bb9b-4e2d-4790-9727-da430d9ec702)
                         *Training process on the CUB 200 dataset*

![Screenshot_3](https://github.com/asimica/txt2img/assets/97536578/d2ddbdaa-15de-4bcb-ac87-320abede7106)
                         *Training process on the Flickr 30k dataset*<br>


The file <code>t5_gan128.py</code> represents a GAN network conditioned on encoded textual captions. The architecture of this network is shown in figure below.<br>                          
![Screenshot_4](https://github.com/asimica/txt2img/assets/97536578/623b2c30-9b52-46ed-919b-68497adc13b6)
The transformer used in this architecture is a pretrained transformer presented in [Exploring the Limits of Transfer Learning with a Unified
Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf), and the GAN network is similar to the one implemented in <code>gan128.py</code>, with aditional inputs for the encoded textual captions.
