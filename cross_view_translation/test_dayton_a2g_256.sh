export CUDA_VISIBLE_DEVICES=0;
python test.py --dataroot ./datasets/dayton_lggan \
	--name dayton_a2g_pretrained \
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
