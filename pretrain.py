
import os
import time
import shutil

import utils
import argparse
from tqdm import trange

import torch

from model import SimCLR_Module

from data_handler import DataHandler_CheXpert_Pretraining
from training_args import add_pretrain_args

import torch.distributed as dist
import torch.multiprocessing as mp

def setup(ngpus_per_node, args):
    args.rank = args.rank * ngpus_per_node + args.device
    dist.init_process_group('nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

def cleanup():
    dist.destroy_process_group()

def main(gpu, ngpus_per_node, args):

    # Distributed Training Setup 
    args.device = gpu
    torch.cuda.device(args.device)
    
    if (args.dist):
        setup(ngpus_per_node, args)
    else:
        assert((ngpus_per_node == 1) and (args.rank == 0))
    
    # Only Output On One Device
    localmaster = (args.rank == 0)

    #################################### Prepare Logger & Output Directory ##################################################

    if (localmaster):
        utils.prepare_directory(args.exp_dir)

        logger_titles = [
            'Lowest Loss',
            
            'Learning Rate',
            'Training Loss',
        ]

        log_dir = os.path.join(args.exp_dir, "training_logs")
        utils.prepare_directory(log_dir)
        logger = utils.Logger(log_dir, 'logs', logger_titles)

    ############################################# Setup DataLoaders #########################################################

    args.batch_size = int(args.batch_size / ngpus_per_node)
    train_dataloader, sampler = DataHandler_CheXpert_Pretraining(args.data_dir).get_dataloaders(args)

    #################################### Load Model, Optimizer, Scheduler & Criterion #######################################
    
    # Load Model
    model = SimCLR_Module(args, verbose=localmaster)
    
    #################################### Begin Finetuning ##################################################################

    start_epoch = 0
    if (localmaster):
        if (args.dist):
            print("\nStarting SimCLR Distributed Pretraining From Epoch %d ..."%(start_epoch))
        else:
            print("\nStarting SimCLR Pretraining From Epoch %d ..."%(start_epoch))
        print("-" * 100)

    lowest_loss = 100
    for epoch in trange(start_epoch, args.n_epochs, desc='epoch_monitor', dynamic_ncols=True):
        
        print('\n')

        # Necessary For Random Batch Resampling
        try:
            sampler.set_epoch(epoch)
        except:
            pass

        train_loss = model.train(train_dataloader, epoch)
        lr = model.get_lr()

        if (localmaster):

            # Remember State Dict
            state = {
                'state_dict': model.simclr.module.state_dict() if (args.dist) else model.simclr.state_dict(),
                'optimizer': model.optimizer.state_dict(),
                'scheduler': model.scheduler.state_dict() if hasattr(model, 'scheduler') else None,
                'epoch': epoch,
            }

            # Save Models
            default_path = os.path.join(args.exp_dir, 'model_last_epoch.ckpt')
            torch.save(state, default_path)

            if (train_loss < lowest_loss):
                shutil.copyfile(default_path, os.path.join(args.exp_dir, 'model_lowest_loss.ckpt'))
                lowest_loss = train_loss

            # Append logger file
            logger.append([
                lowest_loss,
                lr,
                train_loss,
            ], step=epoch)

            print('Training Loss: %.3f'%(train_loss))

    if (localmaster):
        logger.close()
        print("Finetuning Complete. Final Loss: {:.2f}".format(train_loss))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser = add_pretrain_args(parser)
    args = parser.parse_args()

    utils.set_seed(args.seed)

    start_time = time.time() 
    if (args.dist):

        # Find GPUS & Setup Parameters
        ngpus_per_node = torch.cuda.device_count()
        assert (ngpus_per_node >= 2), 'Requires at least 2 GPUs, but found only %d'%(ngpus_per_node)
        args.world_size = ngpus_per_node * args.num_nodes

        mp.spawn(
            main, 
            args=(ngpus_per_node, args), 
            nprocs=ngpus_per_node
        )
    else:
        
        # Find Free GPU
        free_gpu = utils.get_free_gpu()

        main(free_gpu, 1, args)
    
    end_time = time.time() 
    print("\nTotal Time Elapsed: {:.2f}h".format((end_time - start_time) / 3600.0))

