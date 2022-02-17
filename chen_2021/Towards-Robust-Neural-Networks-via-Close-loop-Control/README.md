# Towards-Robust-Neural-Networks-via-Close-loop-Control
This repo contains necessary code for the paper [Towards Robust Neural Networks via Close-loop Control](https://openreview.net/pdf?id=2AL06y9cDE-) 
by [Zhuotong Chen](https://scholar.google.com/citations?user=OVs7TPUAAAAJ&hl=en), [Qianxiao Li](https://discovery.nus.edu.sg/9699-qianxiao-li) 
and [Zheng Zhang](https://web.ece.ucsb.edu/~zhengzhang/).

## Description
The proposed **Close-loop control neural network (CLC-NN)** is a optimal control theory inspried defense method against various perturbations.
It can be applied to any classifier to improve its robustness. Given unknown data, the CLC-NN performes the Pontryagin's Maximum Principle
(PMP) dynamics on the entire state trajectory, the controlled state trajectory is used for final prediction.

### The Controlled structure
![alt text](https://github.com/zhuotongchen/Towards-Robust-Neural-Networks-via-Close-loop-Control/blob/master/assets/Structure.png)

### The demonstration of controlling both input and hidden states
![alt text](https://github.com/zhuotongchen/Towards-Robust-Neural-Networks-via-Close-loop-Control/blob/master/assets/Demons.png)

## Dependencies
```bash
Python == 3.6.9
Pytorch == 1.5.1
numpy == 1.19.0
```

## Running demonstration
The code in this repo are capable of doing the follwoing tasks:
- Performing standard and robust training with FGSM, PGD, label-smoothing 
([main_train_models.py](https://github.com/zhuotongchen/Towards-Robust-Neural-Networks-via-Close-loop-Control/blob/master/main_train_models.py)).
- Evaluating the model performance against random, FGSM, PGD, CW and Manifold-based perturbations, and the PMP defense performance 
([main_evaluation.py](https://github.com/zhuotongchen/Towards-Robust-Neural-Networks-via-Close-loop-Control/blob/master/main_evaluation.py)).
- Training a set of auto-encoders of all hidden states for a given neural netowrk 
([main_train_encoders.py](https://github.com/zhuotongchen/Towards-Robust-Neural-Networks-via-Close-loop-Control/blob/master/main_train_encoders.py)).

## Running description
1. Train a neural network with a specified training method (main_train_models.py)
2. For the linear defense
   - Generate the linear embedding for all input and hidden states.
   - Search for the optimal learning rate and maximum iteration number for the PMP dynamics (main_evaluation.py --pmp_select_parameters).
   - For evaluation with the defense, select defense_type (None, layer_wise_projection, linear_pmp), perturbation type and magnitude.
3. For the nonlinear defense
   - Train a set of auto-encoders for all input and hidden states (main_train_encoders.py)
   - Search for the optimal learning rate and maximum iteration number for the PMP dynamics (main_evaluation.py --pmp_select_parameters).
   - For evaluation with the defense, select defense_type (None, layer_wise_projection, linear_pmp), perturbation type and magnitude.

## License
This project is licensed under the MIT license - see the [LICENSE](https://github.com/zhuotongchen/Towards-Robust-Neural-Networks-via-Close-loop-Control/blob/master/LICENSE) file for more details.

## Citation
If you use this code for your research, please cite our paper:
```bash
@article{chentowards,
  title={TOWARDS ROBUST NEURAL NETWORKS VIA CLOSE-LOOP CONTROL},
  author={Chen, Zhuotong and Li, Qianxiao and Zhang, Zheng}
}
```

## Contact
Please contact <ztchen@ucsb.edu> or <zhuotongchen@gmail.com> if you have any question on the code.

