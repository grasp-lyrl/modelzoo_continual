# Model Zoo: A Growing "Brain" That Learns Continually

Implementation of [Model Zoo: A Growing "Brain" That Learns Continually](https://arxiv.org/abs/2106.03027)


<p float="center">
  <img src="./ssets/modelzoo.png" height="300" hspace="4"/>
  <img src="./assets/fwd_bwd_transfer.png" height="300" hspace="4"/>
</p>
Model Zoo splits the capacity of the model to mitigate task-competition and 
better exploit the relationship between of tasks. It is a continual learner capable of leveraging past tasks to solve new tasks and make use of new tasks to improve on past tasks.

## Setup:

To install a working environment run:
```
conda env create -f env.yaml
```

Download the `.pkl` files for Mini-imagenet 
([link](https://www.kaggle.com/whitemoon/miniimagenet)) and 
copy the files to `./data/mini_imagenet/`


## Usage

The two key files is `modelzoo.py`. The `-h`
flag can be used to list the argparse arguments. For example to run Model Zoo:

```bash
python modelzoo.py --data_config ./config/dataset/coarse_cifar100.yaml \
                   --hp_config ./config/hyperparam/wrn.yaml \
                   --epochs 100 --replay_frac 1.0
```

To run the continual learning variant of the Model Zoo, add the `--continual` flag. The tasks are presented sequentially with the order prescribed by the data config file.

## Directory Structure

```bash
├── modelzoo.py                   # Implementation of Model Zoo
├── config:                       # Configuration files
│   ├── dataset                    
│   └── hyperparam                  
├── datasets                      # Datasets and Dataloaders
│   ├── build_dataset.py          
│   ├── cifar.py                 
│   ├── data.py                 
│   ├── mini_imagenet.py           
│   ├── mnist.py               
│   ├── modmnist.py           
├── net                           # Neural network architectures
│   ├── build_net.py
│   └── wideresnet.py
│   └── smallconv.py
└── utils                         # Utilities for logging/training
    ├── config.py
    ├── logger.py
    └── run_net.py
```

