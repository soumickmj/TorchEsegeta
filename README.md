# TorchEsegeta
Interpretability and Explainability pipeline for PyTorch

Esegeta (origin: ancient Greek) in Italian means interpreter of sacred texts

### Objective:

> As the use of AI based apps and tools is incresing in critical domains like Medicine, Life support, Manufacturing and as well as in common house holds, its very important now that we strip off the black box nature of the machine learning models. The goal of this development is to create a complete package for testing different inerpretability and explainability methods on any given deep learning model. The package should work for both semantic segmemnation model as well as for classification models.

### System Requirements

> The implemented methods here are very resourse heavy and they need brute computaion power. Hence CUDA enabled GPU is bare minimum requirement. Although we are working to make this package work for CPUs as well. But some of the methods may not work or may take very long time with CPU.


### Software Requiements
> The following packages (with dependencies) needed to be installed with ur python packages for this library to work

* numpy
* matplotlib
* torch
* torchio
* tochvision
* Captum [[1]](#1)
* tochray [[2]](#2)
* lime [[6]](#6)
* torch-lucent [[3]](#3)
* CNN Visualization [[4]](#4)
 
`
It is recommended to use conda distribution with python 3.6 environment, Torch 1.6 with Cuda 10.2
`


> To install torchray you must have python cocoapi installaed. This installation is not straight forward with pip. Please follow the following link for cocoapi installation. and then install torch ray.

> [Python COCO Tool API](https://github.com/philferriere/cocoapi)



### Implemented Methods:

>The library is continuously evolving. In first phase we will ensemble methods from other third party libraries. And next we will implement our own methods to augment the capability of this package.
As of now the implemented methods are:

* Captum
    1. Saliency
    2. Integrated gradients
    3. Feature ablation
    4. Guided Backpropagation
    5. Deep lift
    6. De-convolution
    7. Guided grad-cam
    8. Layer activation
    9. Layer conductance
    10. Layer grad shap
    11. Gradient shap
    12. Internal influence
    13. Inpt X-gradient
    14. Deep lift shap
    15. Layer gradient X-Activation
    16. Layer Deep lift
    17. Layer Grad Cam
    18. Shaley Value Sampling

`
Shaley value sampling takes lot of time for image input. Not recommended to perform on image data.
`
    
* CNN Visualization
    1. Guided Backpropagation
    2. Integrated Gradients
    3. Guided grad cam
    4. Score cam
    5. Vanilla Backpropagation
    6. Grad Cam
    7. Grad Image Times
    8. Layer Activation Guided Backpropagation
    9. Layer Visualization
    10. Deep Dream


* Torch Ray
    1. Excitation backpropagation
    2. Contrastive excitation backpropagation
    3. Rise
    4. Deconv
    5. Grad Cam
    6. Gradient
    7. Guided Backpropagation
    8. Linear Approx.



* Lucent
    1. Render visualization
    

* Lime
    1. Lime segmentation- Using Image explainer

>More information regarding the implemented methods and their corresponding parameters are available in: https://github.com/soumickmj/TorchEsegeta/blob/master/EsegetaMethodInfo.pdf

### Additional Features:

* Exception Handling
* Logging
* Side by side visualization of multiple target attributions
* Extended for 3d models
* Timeout for long running methods
* Multi GPU and multi threading
* Automatic mixed precision support
* Extended for patch based models.

## How to use:
> This pipeline is built on top of third party libraries, hence this pipeline is affected/effected by the underlying implementation, complexity and constraints of those. We tried generalize as most as we can. But still, configuring this pipeline itself presents a challenge to the user. But it is a one time activity and worth the pain, as this can produce output from various methods from various libraries. Hence follow the below instructions minutely. For example check the pipelineTester.py file and run it as is with providing the 2D segmentation configuration json file, marking any method use as True.

#### A) **Model declaration**:  

> For classification task the model can be given as it is. but for segmentation task models's forward function needs to be updated with few lines of custom wrapper code so that the output becomes batch*totalClass.  for example check the default section of pipeline.py and modelWrapper.py.
As of now the wrapper_fnction argument of interpret_explain function from Pipeline class takes two values "threshold_based" (default) and "multi_class" for threshold based segmemnation models and for multiclass semantic segmentation models. You need to pass this argument based on the model. The Classification models do not required any specific wrapper if not a very special case. 
Alternatively you can also mention wrapper function separately. those function need to be declared in the methods.py file under Interpretability class. and in configuration json file mark the 'aux_func_flag' as true. This alternate method doesn't work for all the methods, due to constraints in underlying libraries. Or you can personally add more wrappers in sermentWrapper.py file and update the dictionary maintained in pipeline.py file.

#### B) **Input selection**:

> As of now test it with sigle input only. In future version we will enable the multi input feature. This is done to take away some complexity of the pipeline. Once we are happy with its performance , this functionality can be added with minor changes.
Additionally if any sort of zooming required (for non patch based testing), this needs to be handled in the pipeline.py file for variable inp or can be manually done before giving it to the pipeline. And in the configuration json file attributes 'isDepthFirst' and 'is_3d' needs to be mentioned as per model requirement.

> Use Pipeline class or Pipeline multi thread class as per your requirement

#### C) **Configuration File**

> Pass the path for the desired configuration file in pipeline-tester.py. For your help this library contains four sample configuration json files, 
two for 2d models and rest two for 3d models for both Segmentation and Classification task.

>Also see the confuguration file tag descriptions for understanding.

#### D) **Tag descriptions**

<details>
<summary>Tag summary :</summary>

`is_3d :` Make it True if it is a 3D model.

`isDepthFirst :` Make it True if the model accepts input in depth first manner. Make sure to provide input in the same manner.

`batch_dim_present :` : True or False depending on whether batch dimension is present in input data.

`default :` : Generalle keep it False. Only make this true if you want to test with default models provided with this library. Althoug only 2d Segmentation will work as other default models require weight checkpoint, which is not provided along with the library as they huge in size and proprietary.

`dataset :` Mention your dataset name.

`test_run :` Run reference number.

`patch_overlap : ` If using patcher, this the overlap pixel count between patches. Otherwise make it 0

`patch_size :`  if using patcher, this is the patch size. This must be set to -1 if patcher is not used.

`amp_enbled :` Make it True to use Automatic mixed precision.

`share_gpu_threads` Number of methods runnign on same gpu per thread.

`timeout_enabled` Make it true if you want to enable timeout functionality for long running methods. (Only for Linux)

`log_level :` Level of information you want to see in the log file.

`uncertainity_metrics :` Make this flag False for all methods as of now. This is for a future functionality. Making it True may generate unexpected details.

`uncertainity_cascading :`Make this flag 0 for all methods. This is for a future functionality. Making it True may generate unexpected details.

For all other method related tags please check the documentation from the mentioned library.(Mentioned in references)

</details>


### Upcoming Features:

* Handling multiple inputs at time
* Uncertainty/Evaluation methods for generated attributions

## Some generated samples
|2D classification-Lucent            |  2D Segmentation-Captum GBP              |  3D Segmentation-CNN Vis Vanilla BackProp|
|:-------------------------:|:-------------------------:|:-------------------------:|
![image](https://drive.google.com/uc?export=view&id=1W_jbKOVEGseOzi_4o6ZjhH6Tzm437TQS) | <img src="https://drive.google.com/uc?export=view&id=1aPsxHign8GBWPcU9cof09QuWn--ms6X_"  width="300" height="180"> | <img src="https://drive.google.com/uc?export=view&id=1LHOZX9LJ9n-5WksjnSCvgT2-qZl_zcUh"  width="250" height="200"> |



## References
<a id="1">[1]</a> 
Captum : https://captum.ai/

<a id="2">[2]</a>
Torchray : https://github.com/facebookresearch/TorchRay 

<a id="3">[3]</a>
Lucent: https://github.com/greentfrapp/lucent

<a id="4">[4]</a>
CNN Visualization : https://github.com/utkuozbulak/pytorch-cnn-visualizations

<a id="5">[5]</a>
DS6 Paper: https://arxiv.org/pdf/2006.10802.pdf

<a id="6">[6]</a>
Lime: https://github.com/marcotcr/lime

## Credits

If you like this repository, please click on Star!

If you use any of our approaches in your research or use codes from this repository, please cite one of the following (or both) in your publications:

TorchEsegeta pipeline, including the methods for Segmentation:-

> [Soumick Chatterjee, Arnab Das, Chirag Mandal, Budhaditya Mukhopadhyay, Manish Vipinraj, Aniruddh Shukla, Rajatha Nagaraja Rao, Chompunuch Sarasaen, Oliver Speck, Andreas Nürnberger: TorchEsegeta: Framework for Interpretability and Explainability of Image-based Deep Learning Models (Preprint:- arXiv:2110.08429, Oct 2021, Published:- Applied Sciences, Feb 2022)](https://www.mdpi.com/2076-3417/12/4/1834)

BibTeX entry:

```bibtex
@article{chatterjee2022torchesegeta,
  author = {Chatterjee, Soumick and Das, Arnab and Mandal, Chirag and Mukhopadhyay, Budhaditya and Vipinraj, Manish and Shukla, Aniruddh and Nagaraja Rao, Rajatha and Sarasaen, Chompunuch and Speck, Oliver and Nürnberger, Andreas},
title = {TorchEsegeta: Framework for Interpretability and Explainability of Image-Based Deep Learning Models},
journal = {Applied Sciences},
volume = {12},
year = {2022},
number = {4},
article-number = {1834},
url = {https://www.mdpi.com/2076-3417/12/4/1834},
issn = {2076-3417},
doi = {10.3390/app12041834}
}


```
This was also presented as an abstract at ISMRM 2021:
> [Soumick Chatterjee, Arnab Das, Chirag Mandal, Budhaditya Mukhopadhyay, Manish Bhandari Vipinraj Bhandari, Aniruddh Shukla, Oliver Speck, Andreas Nürnberger: Interpretability Techniques for Deep Learning based Segmentation Models (ISMRM May 2021)](https://www.researchgate.net/publication/349589153_Interpretability_Techniques_for_Deep_Learning_based_Segmentation_Models)

Initial version of TorchEsegeta for Classification models can be cited using:-

> [Soumick Chatterjee, Fatima Saad, Chompunuch Sarasaen, Suhita Ghosh, Rupali Khatun, Petia Radeva, Georg Rose, Sebastian Stober, Oliver Speck, Andreas Nürnberger: Exploration of Interpretability Techniques for Deep COVID-19 Classification using Chest X-ray Images (arXiv:2006.02570, June 2020)](https://arxiv.org/abs/2006.02570)

BibTeX entry:

```bibtex
@article{chatterjee2020exploration,
  title={Exploration of interpretability techniques for deep covid-19 classification using chest x-ray images},
  author={Chatterjee, Soumick and Saad, Fatima and Sarasaen, Chompunuch and Ghosh, Suhita and Khatun, Rupali and Radeva, Petia and Rose, Georg and Stober, Sebastian and Speck, Oliver and N{\"u}rnberger, Andreas},
  journal={arXiv preprint arXiv:2006.02570},
  year={2020}
}
```

Thank you so much for your support.
