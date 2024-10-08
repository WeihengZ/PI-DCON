# PI-DCON

This repository is the official implementation of the paper: [Physics-informed discretization-independent deep compositional operator network](https://www.sciencedirect.com/science/article/abs/pii/S0045782524005309), published in Journal of Computer Methods in Applied Mechanics and Engineering. The arxiv version of paper can also be found [here](https://arxiv.org/html/2404.13646v1).

Our research explores physics-informed machine learning methods for **irregular domain geometries** where PDE parameters are represented as a set of function value observations of any size, making the approach **discretization-independent**. In this paper, we introduce the first neural operator model that is discretization-independent and capable of addressing this scenario. However, the model architecture we propose in this paper is intentionally kept straightforward, and we encourage researchers to explore and develop more advanced architectures to further enhance this approach. 

## Overview

If you're interested in using our well-trained model, please refer to the **"User mode"** section. For those with similar research interests looking to explore more advanced model architectures and training algorithms, please check the **"Developer mode"** section. This work is also one of our work for developing [**Neural Operators as Foundation Model for solving PDEs**](https://github.com/WeihengZ/Physics-informed-Neural-Foundation-Operator). Please feel free to check it out as well if you are interested! We are excited to see more and more interesting ideas coming out for this research goal!

## User mode

If you want to reproduce the results of our paper, please first download our [dataset](https://drive.google.com/drive/folders/10c5BWVvd-Oj13tMGhE07Tau07aTWfOhM?usp=sharing) into the folder named "data", and download our [well-trained models](https://drive.google.com/drive/folders/1NFUTkvSoubaTnrcjHf0R29J66E-uJEBz?usp=sharing) into the folder named "res/saved_models". 

After preparing all the data and well-trained models, you need to first install all the required python package (with the Python>=3.8 is preferred) by
```
pip install -r requirements.txt
```

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

## Developer mode

This repository is user-friendly for developing new model architecture of the neural operator model. You can simply explore your self-designed model architecture by the following steps:
* open the script of the Main/models.py, input your self-defined model architectur into the function class of "New_model_darcy" and "New_model_plate".
* Adjust the hyper-parameters in the file "configs/self_defined_Darcy_star.yaml" and "configs/self_defined_plate_dis_high.yaml". 
* Train your model by simply run the following command:
```
cd Main
python exp_pinn_darcy.py --model='self_defined' --phase='train'
python exp_pinn_plate.py --model='self_defined' --phase='train'
```

If you are interested in developing more advanced training algorithms, please check our the script "darcy_utils.py" and "plate_utils.py".

**If you think that the work of the PI-DCON is useful in your research, please consider citing our paper in your manuscript:**
```
@article{zhong2024physics,
  title={Physics-informed discretization-independent deep compositional operator network},
  author={Zhong, Weiheng and Meidani, Hadi},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={431},
  pages={117274},
  year={2024},
  publisher={Elsevier}
}
```
