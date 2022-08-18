
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
import utils
from tqdm import tqdm
import torchvision.models as models

import numpy as np

from libauc.optimizers import PESG
from libauc.losses import AUCMLoss

from data_handler import DataHandler_CovidX_GradCAM

from resnet import get_resnet

def main(gpu, data_dir, kfold_exp_dir):

    # Distributed Training Setup 
    device = gpu
    
    # Only Output On One Device
    localmaster = True

    ############################################# Setup DataLoaders #########################################################

    train_dataloader, test_dataloader, val_dataloader = DataHandler_CovidX_GradCAM(
        data_dir, 
        verbose=localmaster
    ).get_dataloaders()

    #################################### Load Model, Optimizer, Scheduler & Criterion #######################################
    
    trust_scores = []

    for fold in range(5):

        # Load Model
        model, _= get_resnet(depth=50, width_multiplier=1, sk_ratio=0)
            
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, (model.fc.in_features // 2)),
            nn.ReLU(),
            nn.Linear((model.fc.in_features // 2), 1)
        )
        
        model_pth = os.path.join(kfold_exp_dir, 'model_best_acc.ckpt')
        checkpoint = torch.load(model_pth, map_location='cuda:%d'%(device))
        model.load_state_dict(checkpoint['state_dict'])
        threshold = checkpoint['threshold']
        #threshold = 0.5

        model = model.cuda()
        
        print(model.fc)

    #################################### Begin Testing ##################################################################
        
        # Switch To Evaluation Mode
        model.eval()

        print('\nUsing Threshold : %.3f'%(threshold))

        pbar = tqdm(desc='Test Loop', total=len(test_dataloader.dataset), dynamic_ncols=True)

        negatives = []
        n_negatives = []
        positives = []
        n_positives = []

        with torch.no_grad():
            for batch_idx, (_, _, input, target) in enumerate(test_dataloader):

                input = input.to(device)
                target = target.to(device, non_blocking=True)

                # compute output
                output = model(input, apply_fc=True)
                output = torch.sigmoid(output)
                if (output < threshold):
                    negatives.append(output.item())
                    n_negatives.append((target == 0))
                else:
                    positives.append(output.item())
                    n_positives.append((target == 1))

                pbar.update(input.size(0))
        pbar.close()

        negatives = np.array(negatives)
        n_negatives = np.array(n_negatives)
        positives = np.array(positives)
        n_positives = np.array(n_positives)

        ######## Scale Negatives #########

        old_min = min(negatives)
        old_max = max(negatives)

        new_min = 0.0
        new_max = 0.5
        
        old_range = old_max - old_min
        new_range = new_max - new_min

        negatives = ((negatives - old_min) * new_range / old_range) + new_min

        ####### Scale Positives #########

        old_min = min(positives)
        old_max = max(positives)

        new_min = 0.5
        new_max = 1.0

        old_range = old_max - old_min
        new_range = new_max - new_min

        positives = ((positives - old_min) * new_range / old_range) + new_min

        #################################
        
        confidences = []
        
        for i in range(positives.shape[0]):
            score = positives[i]
            correct = n_positives[i]

            if (correct):
                confidences.append(score) 
            else:
                confidences.append(1 - score)

        for i in range(negatives.shape[0]):
            score = negatives[i]
            correct = n_negatives[i]

            if (correct):
                confidences.append(1 - score) 
            else:
                confidences.append(score)

        trust_scores.append(np.mean(confidences))

    with open(os.path.join(kfold_exp_dir, 'trust_scores80.txt'), 'w') as file:
        file.write('Mean : %.5f | STD : %.5f | Percent : %.3f'%(
            np.mean(trust_scores),
            np.std(trust_scores),
            (np.std(trust_scores) / np.mean(trust_scores)) * 100
        ))

if __name__ == "__main__":

    utils.set_seed(123)
    start_time = time.time() 
    
    data_dir = '../archive'
    kfold_exp_dir = 'test80'

    # Find Free GPU
    free_gpu = utils.get_free_gpu()
    main(free_gpu, data_dir, kfold_exp_dir)

    end_time = time.time() 
    print("\nTotal Time Elapsed: {:.2f}h".format((end_time - start_time) / 3600.0))

