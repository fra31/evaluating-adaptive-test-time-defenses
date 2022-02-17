# AID-Purifier: A Light Auxiliary Network for Boosting Adversarial Defense

This repository is the official implementation of 'AID-Purifier: A Light Auxiliary Network for Boosting Adversarial Defense'.

#### Abstract

In this study, we propose _AID-purifier_ that can boost the robustness of adversarially-trained networks by purifying their inputs. AID-purifier is an auxiliary network that works as an add-on to an already trained main classifier. To keep it computationally light, it is trained as a discriminator with a binary cross-entropy loss. To obtain additionally useful information from the adversarial examples, the architecture design is closely related to the information maximization principle where two layers of the main classification network are piped into the auxiliary network. To assist the iterative optimization procedure of purification, the auxiliary network is trained with AVmixup. AID-purifier can be also used together with other purifiers such as PixelDefend for an extra enhancement. Because input purification has been studied relative less when compared to adversarial training or gradient masking, we conduct extensive attack experiments to validate AID-purifier's robustness. The overall results indicate that the best performing adversarially-trained networks can be enhanced further with AID-purifier. 
## Requirements

#### Environment setup
Caution: This version is set for RTX2080ti. If you use other types of GPUs, change the version of torch and others.

To install conda environment with requirements (need to install conda first!):

```install
conda env create --file environment.yml
```
To activate conda environment:
```setup
conda activate AID
```

#### Dataset
If you want to run the code on TinyImageNet, you need to download it to [dataset path].
Other datasets (SVHN, CIFAR-10, CIFAR-100) will be automatically downloaded to [dataset path] when run the below command.

Default [dataset path] is './dataset'. If you download the dataset to another directory, you need to add an argument (-dataset_path [dataset path]) to the below command.

#### Download pre-trained models

You can download the pre-trained main classification networks here:
- [Main classification networks](https://drive.google.com/file/d/10Bj_wptfQ1zSgrRPBwnbTECjgCpYFRjW/view?usp=sharing) trained by (Natural, Madry, Zhang, Lee) on various datasets (SVHN, CIFAR-10, CIFAR-100, TinyImageNet).

Download the files, unzip the folder (main_classification_models folder) and move them to [pth models path].

Default [pth models path] is './pth_models'. If you unzip the folder to another directory, you need to add an argument (-pthpath [pth models path]) to the below command.


## Training + evaluation

To train the AID-Purifier (auxiliary network) in the paper, run this command:

For SVHN
```train
python main.py -mc [base main classifer] -ds svhn -layer2_off -layer3_off -tr_eps 0.047 -tr_alpha 0.008 -def_eps 0.047 -def_alpha 0.012 -gamma 2.0 -bcd 0 -acd 2
```
For CIFAR-10
```train
python main.py -mc [base main classifer] -ds cifar10 -layer2_off -layer4_off -tr_eps 0.031 -tr_alpha 0.004 -def_eps 0.031 -def_alpha 0.008 -gamma 1.5
```
For CIFAR-100
```train
python main.py -mc [base main classifer]] -ds cifar100 -layer2_off -layer4_off -tr_eps 0.031 -tr_alpha 0.008 -def_eps 0.062 -def_alpha 0.008 -gamma 1.5
```
For TinyImageNet
```train
python main.py -mc [base main classifer] -ds tiny -layer1_off -layer4_off -tr_eps 0.062 -tr_alpha 0.004 -def_eps 0.031 -def_alpha 0.008 -gamma 1.5
```

[base main classifer] : e.g. nat, madry, zhang, lee.


## Evaluation only (discriminator model file is needed)

To evaluate, run:

For SVHN
```eval
python main.py -mc [base main classifer] -ds svhn -tt test -layer2_off -layer3_off -def_eps 0.047 -def_alpha 0.012 -bcd 0 -acd 2
```
For CIFAR-10
```eval
python main.py -mc [base main classifer] -ds cifar10 -tt test -layer2_off -layer4_off -def_eps 0.031 -def_alpha 0.008 -def_step 10
```
For CIFAR-100
```eval
python main.py -mc [base main classifer] -ds cifar100 -tt test -layer2_off -layer4_off -def_eps 0.062 -def_alpha 0.008 -def_step 20
```
For TinyImageNet
```eval
python main.py -mc [base main classifer] -ds tiny -tt test -layer1_off -layer4_off -def_eps 0.031 -def_alpha 0.008 -def_step 20
```

Caution: It would be best if you were patient not only for training but also for evaluation.


## Pre-trained models
You can download pretrained the pre-trained AID-Purifier networks here:
- [pre-trained AID-Purifier networks](https://drive.google.com/file/d/1fDvoQ0cfCdB6kLmqNUY1edI0U-wkjcx8/view?usp=sharing) trained on various datasets (SVHN, CIFAR-10, CIFAR-100, TinyImageNet)
  with the Madry main classification network.

Download the files, unzip the folder (purifier_models folder) and move them to [pth models path].


## Results

Our model achieves the following performance under the worst white-box attack on SVHN dataset :


| Model name   | Before purification  | After purification  |
|--------------|----------------------| --------------------|
| Madry        |     22.63%           |      49.85%         |



## Contributing
[MIT LICENSE](./LICENSE)
