export CUDA_VISIBLE_DEVICES=0;
python train.py --dataroot ./datasets/dayton_lggan \
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
