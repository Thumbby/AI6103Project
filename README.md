# NeRF Re-implementation

This project is for AI6103 Deep Learning course project in Nanyang Technological University. 

The group members are:

- CHEN LEI
- LI KAIYU
- HE YUXUAN
- WAN ZHANG

## Installation

```
git clone https://github.com/Thumbby/deeplearning
cd deeplearning
pip install -r requirements.txt
```


  ## Dependencies
The projects is conducted under 2080ti, with the following packages

  - Python == 3.10.6
  - PyTorch
  - matplotlib == 3.6.2
  - numpy == 1.23.4
  - imageio = 2.22.4
  - imageio-ffmpeg == 0.4.7
  - configargparse == 1.5.3
  - tqdm == 4.64.1
  - opencv-python == 4.6.0.66

Every packages will be installed as the requirements.txt

## How To Run?

Before all the operation, you should make sure that you are under the base directory

#### Download the dataset

```
bash download_example_data.sh
```

#### Train with the dataset

Create the *result* directory under the *log* directory, then input the following command in the console:

```
python train.py --config configs/lego.txt --expname result
```

You can check the result in the same directory, also you can change the expname

#### Set the configuration

Every configuration can be set in the *configs/lego.txt*, also you can create a new configuration .txt as long as you use the right one after the *--config* command. Please read the configuration.py to look up the configuration you need.

