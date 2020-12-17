"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i['label']),
                                   # ('synthesized_image', trainer.get_latest_generated()),
                                   # ('global_image', trainer.get_global_generated()),
                                   # ('local_image', trainer.get_local_generated()),
                                   # ('label_3_0', trainer.get_label_3_0_generated()),
                                   # ('label_3_1', trainer.get_label_3_1_generated()),
                                   # ('label_3_2', trainer.get_label_3_2_generated()),
                                   # ('label_3_3', trainer.get_label_3_3_generated()),
                                   # ('label_3_4', trainer.get_label_3_4_generated()),
                                   # ('label_3_5', trainer.get_label_3_5_generated()),
                                   # ('label_3_6', trainer.get_label_3_6_generated()),
                                   # ('label_3_7', trainer.get_label_3_7_generated()),
                                   # ('label_3_8', trainer.get_label_3_8_generated()),
                                   # ('label_3_9', trainer.get_label_3_9_generated()),
                                   # ('label_3_10', trainer.get_label_3_10_generated()),
                                   # ('label_3_11', trainer.get_label_3_11_generated()),
                                   # ('label_3_12', trainer.get_label_3_12_generated()),
                                   # ('label_3_13', trainer.get_label_3_13_generated()),
                                   # ('label_3_14', trainer.get_label_3_14_generated()),
                                   # ('label_3_15', trainer.get_label_3_15_generated()),
                                   # ('label_3_16', trainer.get_label_3_16_generated()),
                                   # ('label_3_17', trainer.get_label_3_17_generated()),
                                   # ('label_3_18', trainer.get_label_3_18_generated()),
                                   # ('label_3_19', trainer.get_label_3_19_generated()),
                                   # ('label_3_20', trainer.get_label_3_20_generated()),
                                   # ('label_3_21', trainer.get_label_3_21_generated()),
                                   # ('label_3_22', trainer.get_label_3_22_generated()),
                                   # ('label_3_23', trainer.get_label_3_23_generated()),
                                   # ('label_3_24', trainer.get_label_3_24_generated()),
                                   # ('label_3_25', trainer.get_label_3_25_generated()),
                                   # ('label_3_26', trainer.get_label_3_26_generated()),
                                   # ('label_3_27', trainer.get_label_3_27_generated()),
                                   # ('label_3_28', trainer.get_label_3_28_generated()),
                                   # ('label_3_29', trainer.get_label_3_29_generated()),
                                   # ('label_3_30', trainer.get_label_3_30_generated()),
                                   # ('label_3_31', trainer.get_label_3_31_generated()),
                                   # ('label_3_32', trainer.get_label_3_32_generated()),
                                   # ('label_3_33', trainer.get_label_3_33_generated()),
                                   # ('label_3_34', trainer.get_label_3_34_generated()),
                                   # ('real_image', data_i['image'])])
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('global_image', trainer.get_global_generated()),
                                   ('local_image', trainer.get_local_generated()),
                                   ('global_attention', trainer.get_global_attention()),
                                   ('local_attention', trainer.get_local_attention()),
                                   # ('label_3_0', trainer.get_label_3_0_generated()),
                                   # ('label_3_1', trainer.get_label_3_1_generated()),
                                   # ('label_3_2', trainer.get_label_3_2_generated()),
                                   # ('label_3_3', trainer.get_label_3_3_generated()),
                                   # ('label_3_4', trainer.get_label_3_4_generated()),
                                   # ('label_3_5', trainer.get_label_3_5_generated()),
                                   # ('label_3_6', trainer.get_label_3_6_generated()),
                                   ('label_3_7', trainer.get_label_3_7_generated()),
                                   ('image_3_7', trainer.get_image_3_7_generated()),
                                   ('label_3_8', trainer.get_label_3_8_generated()),
                                   ('image_3_8', trainer.get_image_3_8_generated()),
                                   # ('label_3_9', trainer.get_label_3_9_generated()),
                                   # ('label_3_10', trainer.get_label_3_10_generated()),
                                   ('label_3_11', trainer.get_label_3_11_generated()),
                                   ('image_3_11', trainer.get_image_3_11_generated()),
                                   # ('label_3_12', trainer.get_label_3_12_generated()),
                                   # ('label_3_13', trainer.get_label_3_13_generated()),
                                   # ('label_3_14', trainer.get_label_3_14_generated()),
                                   # ('label_3_15', trainer.get_label_3_15_generated()),
                                   # ('label_3_16', trainer.get_label_3_16_generated()),
                                   # ('label_3_17', trainer.get_label_3_17_generated()),
                                   # ('label_3_18', trainer.get_label_3_18_generated()),
                                   # ('label_3_19', trainer.get_label_3_19_generated()),
                                   # ('label_3_20', trainer.get_label_3_20_generated()),
                                   ('label_3_21', trainer.get_label_3_21_generated()),
                                   ('image_3_21', trainer.get_image_3_21_generated()),
                                   # ('label_3_22', trainer.get_label_3_22_generated()),
                                   # ('label_3_23', trainer.get_label_3_23_generated()),
                                   # ('label_3_24', trainer.get_label_3_24_generated()),
                                   # ('label_3_25', trainer.get_label_3_25_generated()),
                                   # ('label_3_26', trainer.get_label_3_26_generated()),
                                   ('label_3_27', trainer.get_label_3_27_generated()),
                                   ('image_3_27', trainer.get_image_3_27_generated()),
                                   # ('label_3_28', trainer.get_label_3_28_generated()),
                                   # ('label_3_29', trainer.get_label_3_29_generated()),
                                   # ('label_3_30', trainer.get_label_3_30_generated()),
                                   # ('label_3_31', trainer.get_label_3_31_generated()),
                                   # ('label_3_32', trainer.get_label_3_32_generated()),
                                   ('label_3_33', trainer.get_label_3_33_generated()),
                                   ('image_3_33', trainer.get_image_3_33_generated()),
                                   # ('label_3_34', trainer.get_label_3_34_generated()),
                                   ('real_image', data_i['image'])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
