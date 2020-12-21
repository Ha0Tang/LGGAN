## Contents

  - [Installation](#Installation)
  - [Dataset Preparation](#Dataset-Preparation)
  - [Generating Images Using Pretrained Model](#Generating-Images-Using-Pretrained-Model)
  - [Train and Test New Models](#Train-and-Test-New-Models)
  - [Code Structure](#Code-Structure)
  - [Evaluation](#Evaluation)

## Installation

Clone this repo.
```bash
git clone https://github.com/Ha0Tang/LGGAN
cd LGGAN/
```

This code requires PyTorch 0.4.1 and python 3.6+. Please install dependencies by
```bash
pip install -r requirements.txt (for pip users)
```
or 

```bash
./scripts/conda_deps.sh (for Conda users)
```

To reproduce the results reported in the paper, you would need a NVIDIA GeForce GTX 1080 Ti GPUs with 11 memory.

## Dataset Preparation

For SVA, Dayton or CVUSA, the datasets must be downloaded beforehand. Please download them on the respective webpages. In addition, we put a few sample images in this [code repo](https://github.com/Ha0Tang/LGGAN/tree/master/cross_view_translation/datasets/samples). Please cite their papers if you use the data. 

**Preparing SVA Dataset**. The dataset can be downloaded [here](http://imagelab.ing.unimore.it/imagelab/page.asp?IdPage=19).
Ground Truth semantic maps are not available for this datasets. We adopt [RefineNet](https://github.com/guosheng/refinenet) trained on CityScapes dataset for generating semantic maps and use them as training data in our experiments. Please cite their papers if you use this dataset.
Train/Test splits for SVA dataset can be downloaded from [here](https://github.com/Ha0Tang/LGGAN/tree/master/cross_view_translation/datasets/sva_split).

**Preparing Dayton Dataset**. The dataset can be downloaded [here](https://github.com/lugiavn/gt-crossview). In particular, you will need to download dayton.zip. 
Ground Truth semantic maps are not available for this datasets. We adopt [RefineNet](https://github.com/guosheng/refinenet) trained on CityScapes dataset for generating semantic maps and use them as training data in our experiments. Please cite their papers if you use this dataset.
Train/Test splits for Dayton dataset can be downloaded from [here](https://github.com/Ha0Tang/SelectionGAN/tree/master/selectiongan_v1/datasets/dayton_split).

**Preparing CVUSA Dataset**. The dataset can be downloaded [here](https://drive.google.com/drive/folders/0BzvmHzyo_zCAX3I4VG1mWnhmcGc), which is from this [page](http://cs.uky.edu/~jacobs/datasets/cvusa/). After unzipping the dataset, prepare the training and testing data as discussed in [SelectionGAN](https://arxiv.org/abs/1904.06807). We adopt [RefineNet](https://github.com/guosheng/refinenet) trained on CityScapes dataset for generating semantic maps and use them as training data in our experiments. Please cite their papers if you use this dataset.

**Dataset Pre-processing**. After downloading the dataset, prepare the data like [here](https://github.com/Ha0Tang/LGGAN/tree/master/cross_view_translation/datasets/others/sva), then run the following script to generate the data like [here](https://github.com/Ha0Tang/LGGAN/tree/master/cross_view_translation/datasets/samples) for training and testing:
```
cd scripts/
matlab -nodesktop -nosplash -r "data_preprocessing"
```

We also provide the prepared datasets for your convience.
```
sh datasets/download_lggan_dataset.sh [dataset]
```
where `[dataset]` can be one of `cvusa_lggan`, `dayton_lggan`, `dayton_ablation_lggan`, or `sva_lggan`.

**Preparing Your Own Datasets**. Each training sample in the dataset will contain {Ia,Ig,Sa,Sg,La,Lg}, where Ia=aerial image, Ig=ground image, Sa=color semantic map for aerial image, Sg=color semantic map for ground image, La=semantic label for aerial image, Lg=semantic label for ground image. Of course, you can use LGGAN for your own datasets and tasks.

## Generating Images Using Pretrained Model

Once the dataset is ready. The result images can be generated using pretrained models.

1. You can download a pretrained model (e.g. sva) with the following script:

```
bash ./scripts/download_lggan_model.sh sva
```
The pretrained model is saved at `./checkpoints/[type]_pretrained`. Check [here](https://github.com/Ha0Tang/LGGAN/blob/master/cross_view_translation/scripts/download_lggan_model.sh) for all the available LGGAN models.

2. Generate images using the pretrained model.
```bash
python test.py --dataroot [path_to_dataset] \
	--name [type]_pretrained \
	--model lggan \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm batch \
	--gpu_ids 0 \
	--batchSize [BS] \
	--loadSize [LS] \
	--fineSize [FS] \
	--no_flip \
	--eval
```

`[path_to_dataset]` is the path to the dataset. Dataset can be one of `sva`, `dayton_a2g`, `dayton_g2a` and `cvusa`. `[type]_pretrained` is the directory name of the checkpoint file downloaded in Step 1, which should be one of `sva_pretrained`, `dayton_a2g_64_pretrained`, `dayton_g2a_64_pretrained`, `dayton_a2g_pretrained`, `dayton_g2a_pretrained` and `cvusa_pretrained`. 
If you are running on CPU mode, change `--gpu_ids 0` to `--gpu_ids -1`. For [`BS`, `LS`, `FS`], please see `Train & Test` sections.

Note that testing require large amount of disk space. If you don't have enough space, append `--saveDisk` on the command line.
    
3. The outputs images are stored at `./results/[type]_pretrained/` by default. You can view them using the autogenerated HTML file in the directory.

## Train and Test New Models

New models can be trained and tested with the following commands.

1. Prepare dataset. 

2. Training & Testing

For SVA dataset:
```bash
export CUDA_VISIBLE_DEVICES=0;
python train.py --dataroot ./datasets/sva_local_global 
	--name sva_lggan \
	--model lggan \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 4 \
	--loadSize 286 \
	--fineSize 256 \
	--no_flip \
	--niter 10 \
	--niter_decay 10 \
	--display_id 0

python test.py --dataroot ./datasets/sva_local_global \
	--name sva_lggan \
	--model lggan \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 4 \
	--loadSize 286 \
	--fineSize 256 \
	--no_flip \
	--eval
```

For CVUSA dataset:
```bash
export CUDA_VISIBLE_DEVICES=1;
python train.py --dataroot ./dataset/cvusa_local_global \
	--name cvusa_lggan \
	--model lggan \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 4 \
	--loadSize 286 \
	--fineSize 256 \
	--no_flip \
	--niter 15 \
	--niter_decay 15 \
	--display_id 0

python test.py --dataroot ./dataset/cvusa_local_global \
	--name cvusa_lggan \
	--model lggan \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 4 \
	--loadSize 286 \
	--fineSize 256 \
	--no_flip \
	--eval
```

For Dayton (a2g direction, 256) dataset:
```bash
export CUDA_VISIBLE_DEVICES=6;
python train.py --dataroot ./datasets/dayton_a2g_local_global \
	--name dayton_a2g_lggan \
	--model lggan \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 4 \
	--loadSize 286 \
	--fineSize 256 \
	--no_flip \
	--display_id 0 \
	--niter 20 \
	--niter_decay 15

python test.py --dataroot ./datasets/dayton_a2g_local_global \
	--name dayton_a2g_lggan \
	--model lggan \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 4 \
	--loadSize 286 \
	--fineSize 256 \
	--no_flip \
	--eval
```

For Dayton (g2a direction, 256) dataset:
```bash
export CUDA_VISIBLE_DEVICES=5;
python train.py --dataroot ./datasets/dayton_g2a_local_global \
	--name dayton_g2a_lggan \
	--model lggan \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 4 \
	--loadSize 286 \
	--fineSize 256 \
	--no_flip \
	--display_id 0 \
	--niter 20 \
	--niter_decay 15

python test.py --dataroot ./datasets/dayton_g2a_local_global \
	--name dayton_g2a_lggan \
	--model lggan \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 4 \
	--loadSize 286 \
	--fineSize 256 \
	--no_flip \
	--eval
```

For Dayton (a2g direction, 64) dataset:
```bash
export CUDA_VISIBLE_DEVICES=7;
python train.py --dataroot ./datasets/dayton_a2g_local_global \
	--name dayton_a2g_64_lggan \
	--model lggan \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 16 \
	--loadSize 72 \
	--fineSize 64 \
	--no_flip \
	--display_id 0 \
	--niter 50 \
	--niter_decay 50

python test.py --dataroot ./datasets/dayton_a2g_local_global \
	--name dayton_a2g_64_lggan \
	--model lggan \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 16 \
	--loadSize 72 \
	--fineSize 64 \
	--no_flip \
	--eval
```

For Dayton (g2a direction, 64) dataset:
```bash
export CUDA_VISIBLE_DEVICES=7;
python train.py --dataroot ./datasets/dayton_g2a_local_global \
	--name dayton_g2a_64_lggan \
	--model lggan \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 16 \
	--loadSize 72 \
	--fineSize 64 \
	--no_flip \
	--display_id 0 \
	--niter 50 \
	--niter_decay 50

python test.py --dataroot ./datasets/dayton_g2a_local_global \
	--name dayton_g2a_64_lggan \
	--model lggan \
	--which_model_netG resnet_9blocks \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm instance \
	--gpu_ids 0 \
	--batchSize 16 \
	--loadSize 72 \
	--fineSize 64 \
	--no_flip \
	--eval
```

### Training & Testing Tips

When training, there are many options you can specify. Please use `python train.py --help`. The specified options are printed to the console. To specify the number of GPUs to utilize, use `export CUDA_VISIBLE_DEVICES=[GPU_ID]`. If you are running on CPU mode, change `--gpu_ids 0` to `--gpu_ids -1`.

To view training results and loss plots on local computers, set `--display_id` to a non-zero value and run `python -m visdom.server` on a new terminal and click the URL [http://localhost:8097](http://localhost:8097/).
On a remote server, replace `localhost` with your server's name, such as [http://server.trento.cs.edu:8097](http://server.trento.cs.edu:8097).

When testing, use `--how_many` to specify the maximum number of images to generate. By default, it loads the latest checkpoint. It can be changed using `--which_epoch`. Note that testing require large amount of disk space since LGGAN will generate lots of intermediate results. If you don't have enough disk space, append `--saveDisk` on the testing command line.

### Can I continue/resume my training? 
To fine-tune a pre-trained model, or resume the previous training, use the `--continue_train --which_epoch <int> --epoch_count<int+1>` flag. The program will then load the model based on epoch `<int>` you set in `--which_epoch <int>`. Set `--epoch_count <int+1>` to specify a different starting epoch count.

## Code Structure

- `train.py`, `test.py`: the entry point for training and testing.
- `models/lggan_model.py`: creates the networks, and compute the losses.
- `models/networks/`: defines the architecture of all models for LGGAN.
- `options/`: creates option lists using `argparse` package.
- `data/`: defines the class for loading images and controllable structures.

## Evaluation
Please follow our [SelectionGAN](https://github.com/Ha0Tang/SelectionGAN/tree/master/selectiongan_v1#evaluation-code) to evalute the trained models.