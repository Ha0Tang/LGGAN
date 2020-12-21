export CUDA_VISIBLE_DEVICES=0;
python test.py --dataroot ./datasets/dayton_lggan \
	--name dayton_a2g_64_pretrained \
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
