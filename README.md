# PI-DCON

This repository is the official implementation of the paper: [Physics-informed discretization-independent deep compositional operator network](https://www.sciencedirect.com/science/article/abs/pii/S0045782524005309), published in Journal of Computer Methods in Applied Mechanics and Engineering. The arxiv version of paper can also be found [here](https://arxiv.org/html/2404.13646v1).

## Implementation of our codes

If you want to reproduce the results of our paper, please first download our [dataset](https://drive.google.com/drive/folders/10c5BWVvd-Oj13tMGhE07Tau07aTWfOhM?usp=sharing) into the folder named "data", and download our [well-trained models](https://drive.google.com/drive/folders/1NFUTkvSoubaTnrcjHf0R29J66E-uJEBz?usp=sharing) into the folder named "res/saved_models". 

Then you can evaluate the prediction accuracy of our proposed DCON model in the testing dataset for the darcy problem and 2D plate stress problem with the following commands:
```
cd Main
python exp_pinn_darcy.py --model='DCON' --phase='test'
python exp_pinn_plate.py --model='DCON' --phase='test'
```

Or implement the model training by replacing the "phase" argument:
```
cd Main
python exp_pinn_darcy.py --model='DCON' --phase='train'
python exp_pinn_plate.py --model='DCON' --phase='train'
```

If you want to reproduce the results of the baseline model: DeepONet and Improved architecture of DeepONet, you can simply replace the "model" argument:
```
cd Main
python exp_pinn_darcy.py --model='DON' --phase='train'
python exp_pinn_plate.py --model='DON' --phase='train'
python exp_pinn_darcy.py --model='IDON' --phase='train'
python exp_pinn_plate.py --model='IDON' --phase='train'
```
