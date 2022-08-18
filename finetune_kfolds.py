
import os
import time
import shutil
import pandas as pd

import utils
import argparse
from tqdm import trange

import torch

from model import Resnet50_Module, Densenet121_Module

from data_handler import DataHandler_KFold_CovidX_Finetuning
from training_args import add_kfold_finetune_args

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

    ############################################# Setup K-Fold DataLoaders #########################################################

    args.batch_size = int(args.batch_size / ngpus_per_node)
    k_fold_datahandler = DataHandler_KFold_CovidX_Finetuning(args.data_dir, args.n_folds, verbose=localmaster)
    test_dataloader = k_fold_datahandler.get_test_dataloader(args)

    global_evaluation_metrics = { 'Fold %d'%(k + 1) : {} for k in range(args.n_folds)}

    for fold, train_dataloader, val_dataloader, sampler in k_fold_datahandler.generate_folds(args):
        
        if (localmaster):
            fold_exp_dir = os.path.join(args.exp_dir, 'Fold_%d'%(fold))
            utils.prepare_directory(fold_exp_dir)

            logger_titles = [
                'Best ROC-AUC',
                'Best F1',
                'Learning Rate',
                'Training Loss',
                'Validation Loss',
                'Validation ROC-AUC',
                'Validation F1',
                'Threshold',
            ]

            log_dir = os.path.join(fold_exp_dir, "training_logs")
            utils.prepare_directory(log_dir)
            logger = utils.Logger(log_dir, 'logs', logger_titles)

        # Load Model
        imratio = train_dataloader.dataset.imratio[1]
        train_iters_per_epoch = len(train_dataloader.dataset) // (args.batch_size * ngpus_per_node * args.num_nodes)
        if (args.arch == 'resnet50'):
            model = Resnet50_Module(
                args, 
                verbose=True,
                imratio=imratio,
                train_iters_per_epoch=train_iters_per_epoch,
            )
        elif (args.arch == 'densenet121'):
            model = Densenet121_Module(
                args, 
                verbose=True,
                imratio=imratio,
                train_iters_per_epoch=train_iters_per_epoch,
            )   

        start_epoch = 0
        if (localmaster):
            if (args.dist):
                print("\nStarting %s Distributed Finetuning From Epoch %d ..."%(args.pretrained, start_epoch))
            else:
                print("\nStarting %s Finetuning From Epoch %d ..."%(args.pretrained, start_epoch))
            print("-" * 50)

        lowest_loss = 100
        best_acc1 = 0
        best_AUC = 0
        best_f1 = 0
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
                val_loss, acc1, AUC, average_f1, threshold = model.validate(val_dataloader)

                # Remember State Dict
                state = {
                    'state_dict': model.module.model.state_dict() if (args.dist) else model.model.state_dict(),
                    'optimizer': model.optimizer.state_dict(),
                    'scheduler': model.scheduler.state_dict() if hasattr(model, 'scheduler') else None,
                    'epoch': epoch,
                    'threshold': threshold,
                }

                # Save Models
                default_path = os.path.join(fold_exp_dir, 'model_last_epoch.ckpt')
                torch.save(state, default_path)

                if (acc1 > best_acc1):
                    shutil.copyfile(default_path, os.path.join(fold_exp_dir, 'model_best_acc.ckpt'))
                    best_acc1 = acc1

                if (AUC > best_AUC):
                    shutil.copyfile(default_path, os.path.join(fold_exp_dir, 'model_best_auc.ckpt'))
                    best_AUC = AUC

                if (average_f1 > best_f1):
                    shutil.copyfile(default_path, os.path.join(fold_exp_dir, 'model_best_f1.ckpt'))
                    best_f1 = average_f1

                if (val_loss < lowest_loss):
                    shutil.copyfile(default_path, os.path.join(fold_exp_dir, 'model_lowest_loss.ckpt'))
                    lowest_loss = val_loss

                # Append logger file
                logger.append([
                    best_AUC,
                    best_f1,
                    lr,
                    train_loss,
                    val_loss,
                    AUC,
                    average_f1,
                    threshold,
                ], step=epoch)

                print('Training Loss: %.3f | Validation Loss : %.3f | ROC-AUC : %.3f'%(train_loss, val_loss, AUC))

        if (localmaster):
            logger.close()
            print("Finetuning Complete. Final Loss: {:.2f}".format(train_loss))

            ################################################### Final Evaluation ####################################################

            print('\nCalculating Metrics For Fold %d ...'%(fold))
            print('-' * 50)

            # Create Metrics Directory
            model_names = ['model_last_epoch', 'model_best_acc', 'model_best_auc', 'model_best_f1', 'model_lowest_loss']
            
            global_evaluation_metrics['Fold %d'%(fold)] = {model_name : {} for model_name in model_names}
            
            for model_name in model_names:
                save_dir = os.path.join(fold_exp_dir, 'metrics', model_name)
                utils.prepare_directory(save_dir)
                acc1, precision_scores, recall_scores, f1_scores, AUC = model.evaluate(
                    test_dataloader, 
                    os.path.join(fold_exp_dir, model_name + '.ckpt'), 
                    save_dir
                )

                global_evaluation_metrics['Fold %d'%(fold)][model_name].update({'Top 1 Accuracy' : acc1})
                global_evaluation_metrics['Fold %d'%(fold)][model_name].update({'ROC-AUC' : AUC})

                global_evaluation_metrics['Fold %d'%(fold)][model_name].update({'Precision Negative' : precision_scores[0]})
                global_evaluation_metrics['Fold %d'%(fold)][model_name].update({'Precision Positive' : precision_scores[1]})
                
                global_evaluation_metrics['Fold %d'%(fold)][model_name].update({'Recall Negative' : recall_scores[0]})
                global_evaluation_metrics['Fold %d'%(fold)][model_name].update({'Recall Positive' : recall_scores[1]})

                global_evaluation_metrics['Fold %d'%(fold)][model_name].update({'F1 Score Negative' : f1_scores[0]})
                global_evaluation_metrics['Fold %d'%(fold)][model_name].update({'F1 Score Positive' : f1_scores[1]})

            df = pd.DataFrame.from_dict(global_evaluation_metrics['Fold %d'%(fold)], orient='index')
            save_path = os.path.join(args.exp_dir, 'cross_validation_logs.csv')
            with open(save_path, 'a') as f:
                f.write("\nFold %d"%(fold))
            df.to_csv(save_path, mode='a')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser = add_kfold_finetune_args(parser)
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
