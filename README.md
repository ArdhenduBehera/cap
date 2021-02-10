# Context-aware Attentional Pooling (CAP)
A repository for the code used to create and train the model defined in 'Context-aware Attentional Pooling (CAP) for Fine-grained Visual Classification' from AAAI 2021 (See the ___Paper___ section).
<kbd>![AFLW validation results](doc/Figure_1.JPG?raw=true)</kbd>

## Published Results
<kbd>![AFLW validation results](doc/Table_1.JPG?raw=true)</kbd>
<kbd>![AFLW validation results](doc/Table_2.JPG?raw=true)</kbd>

## Paper
This paper was published in the AAAI 2021 conference. The extended version is available via [ArXiv](https://arxiv.org/abs/2101.06635). A pdf copy has been provided in this repository (AAAI_9885_Behera.pdf).

## Abstract
Deep convolutional neural networks (CNNs) have shown a strong ability in mining discriminative object pose and parts information for image recognition. For fine-grained recognition, context-aware rich feature representation of object/scene plays a key role since it exhibits a significant variance in the same subcategory and subtle variance among different subcategories. Finding the subtle variance that fully characterizes the object/scene is not straightforward. To address this, we propose a novel context-aware attentional pooling (CAP) that effectively captures subtle changes via sub-pixel gradients, and learns to attend informative integral regions and their importance in discriminating different subcategories without requiring the bounding-box and/or distinguishable part annotations. We also introduce a novel feature encoding by considering the intrinsic consistency between the informativeness of the integral regions and their spatial structures to capture the semantic correlation among them. Our approach is simple yet extremely effective and can be easily applied on top of a standard classification backbone network. We evaluate our approach using six state-of-the-art (SotA) backbone networks and eight benchmark datasets. Our method significantly outperforms the SotA approaches on six datasets and is very competitive with the remaining two.

## Creating the Model
### Python Modules
All required imports can be install by running:
```
pip install -r requirements.txt
```
This will install:
* Numpy
* OpenCV
* Tensorflow 1.13.1
* Keras 2.2.4

### Datasets
The following publicly available datasets were used in the study:
* FGVC Aircraft (https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
* Stanford Cars (https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* Stanford Dogs (http://vision.stanford.edu/aditya86/ImageNetDogs/)
* CUB (Caltech-UCSD Birds-200-2011) (http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
* Oxford Flowers102 (https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
* Oxford Pets (https://www.robots.ox.ac.uk/~vgg/data/pets/)
* NABirds (https://dl.allaboutbirds.org/nabirds)
* Food-101 (https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

### Running the Code
The model can be compiled and trained by running the following command:
```
python train_CAP.py
```
Once the model is compiled 2 folders will be created; Metrics and TrainedModels. These are where the evaluation results and model checkpoints will be saved respectively. The other scripts are supplementary to this training script and do not need to be used directly.
Any parameters such as batch size, learning rate, or the path to the dataset can be changed on lines 24-37 of train_CAP.py.

We found each dataset performed best with a specific set of parameters:
| Dataset  | Batch Size | Learning Rate | 
| -------- | ---------- | ------------- |
| Aircraft | 4          | 1e-6          |
| Cars     | 4          | 1e-5          |
| Dogs     | 12         | 1e-6          |
| CUB      | 8          | 1e-6          |
| Flowers  | 4          | 1e-6          |
| Pets     | 12         | 1e-6          |

## Citation
Please include the following citation in any publications using this work:
```
Behera, A., Wharton, Z., Hewage, P., Bera, A., 2021. Context-aware Attentional Pooling (CAP) for Fine-grained Visual Classification. In: 35th AAAI Conference on Artificial Intelligence, 2-5 Feb 2021.
```
