import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()

    opt.dataset_mode = 'aligned'         # force dataset to be aligned
    opt.model = 'pix2pix'                # ensure pix2pix model
    opt.direction = 'AtoB'               # set direction
    opt.display_id = -1                  # disable visdom if not used

    dataset = create_dataset(opt)        # returns AlignedDataset
    dataset_size = len(dataset)
    print('✅ Number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        model.update_learning_rate()

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

            if total_iters % opt.save_latest_freq == 0:
                print(f'Saving latest model (epoch {epoch}, total_iters {total_iters})')
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print(f'Saving model at end of epoch {epoch}')
            model.save_networks('latest')
            model.save_networks(epoch)

        print(f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {int(time.time() - epoch_start_time)} sec')
