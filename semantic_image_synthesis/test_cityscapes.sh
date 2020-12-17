python train.py --name LGGAN_cityscapes --dataset_mode cityscapes --dataroot ./datasets/Cityscapes/data --niter 100 --niter_decay 100 --gpu_ids 0,1,2,3 --checkpoints_dir ./checkpoints --no_l1_loss --batchSize 8;
python test.py --name LGGAN_cityscapes --dataset_mode cityscapes --dataroot ./datasets/Cityscapes/data --gpu_ids 0 --results_dir ./results --checkpoints_dir ./checkpoints --batchSize 1;