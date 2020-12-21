export CUDA_VISIBLE_DEVICES=0;
python train.py --dataroot ./datasets/sva_lggan \
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
