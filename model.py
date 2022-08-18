import timm
import os
import math
import utils
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision.models as models

from libauc.optimizers import PESG
from libauc.losses import AUCMLoss 

from resnet import get_resnet

def linear_warmup_decay(warmup_steps, total_steps, cosine=True, linear=False):
    """Linear warmup for warmup_steps, optionally with cosine annealing or linear decay to 0 at total_steps."""
    assert not (linear and cosine)

    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        if not (cosine or linear):
            # no decay
            return 1.0

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        if cosine:
            # cosine decay
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # linear decay
        return 1.0 - progress

    return fn

class Base_Module:
    def __init__(self, args, verbose):
        
        self.args = args
        self.optim = args.optim
        self.dist = args.dist
        self.device = args.device
        self.pretrained = args.pretrained

        self.n_output = 1
        self.lr_intervals = [50, 75] if (args.n_epochs == 100) else [args.n_epochs // 2]
        self.verbose = verbose
        self.model = None

    def construct_optimizer(self, imratio, train_iters_per_epoch=None):

        # Construct Optimizer & Criterion
        if (self.optim == 'SGD'):
            self.criterion = nn.BCEWithLogitsLoss() if (self.n_output == 1) else nn.CrossEntropyLoss()
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=self.args.weight_decay,
            )
          
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.n_epochs)

            '''
            if ((train_iters_per_epoch == 0) and (self.verbose)):
                print('[Warning] Iterations per epoch is zero')

            warmup_steps = train_iters_per_epoch * self.args.warmup_epochs
            total_steps = train_iters_per_epoch * self.args.n_epochs

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True)
            )
            '''
            
        elif (self.optim == 'AUC'):

            if ((imratio == 0) and (self.verbose)):
                print('[Warning] Imratio is zero')

            self.criterion = AUCMLoss(imratio=imratio, device=self.device)
            self.optimizer = PESG(
                self.model,
                a=self.criterion.a,
                b=self.criterion.b,
                alpha=self.criterion.alpha,
                imratio=imratio,
                lr=self.args.lr,
                gamma=self.args.gamma,
                margin=self.args.margin,
                weight_decay=self.args.weight_decay,
                device=self.device
            )

        else:
            raise NotImplementedError

        if (self.verbose):
            print('\nUsing %s Optimizer With %.4f Learning Rate | Imratio : %.4f'%(self.optim, self.args.lr, imratio))

    def get_lr(self):
        if (self.optim == 'SGD'):
            return self.scheduler.get_last_lr()[-1]
        elif (self.optim == 'AUC'):
            return self.optimizer.lr

    # Train
    def train(self, dataloader, epoch):

        losses = utils.AverageMeter()

        # Monitor Uniform Sampler
        class_prob_0 = utils.AverageMeter()
        class_prob_1 = utils.AverageMeter()

        # Switch To Train Mode
        self.model.train()

        _len = len(dataloader) * dataloader.batch_size if (self.dist) else len(dataloader.dataset) 

        if (self.verbose):
            pbar = tqdm(desc='Train Loop', total=_len , dynamic_ncols=True)
    
        # Update Learning Rate (AUC) 
        if ((self.optim == 'AUC') and (epoch in self.lr_intervals)):
            self.optimizer.update_regularizer(decay_factor=10)

        for input, target in dataloader:

            # Add To Monitor
            class_prob_0.update(torch.sum(target == 0) / target.size(0))
            class_prob_1.update(torch.sum(target == 1) / target.size(0))

            input = input.cuda(self.device)
            target = target.cuda(self.device) 

            # Compute Output
            output = self.forward(input)          
            if (self.optim == 'AUC'):
                output = torch.sigmoid(output)
            output = output.reshape(target.shape)
            target = target.type(torch.float)

            # Compute Loss
            loss = self.criterion(output, target)

            # Record Loss (Updates Average Meter With The Accuracy Value)
            losses.update(loss.item(), input.size(0))

            # Compute Gradient And Do Optimizer Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (self.verbose):
                pbar.update(input.size(0))

        # Update Learning Rate (SGD) 
        if (self.optim != 'AUC'):
          self.scheduler.step()

        if (self.verbose):
            pbar.close()    
            print('\nEpoch [ %d / %d ] | Class Probabilities [ %.2f | %.2f ]'%(
                epoch, self.args.n_epochs, class_prob_0.avg, class_prob_1.avg))

        return losses.avg

    # Validate 
    def validate(self, dataloader):

        losses = utils.AverageMeter()

        outputs = []
        targets = []

        # Switch To Evaluation Mode
        self.model.eval()

        if (self.verbose):
            pbar = tqdm(desc='Valid Loop', total=len(dataloader.dataset), dynamic_ncols=True)
        
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(dataloader):
                
                input = input.cuda(self.device)
                target = target.cuda(self.device)

                # Compute Output
                output = self.forward(input)
                if (self.optim == 'AUC'):
                    output = torch.sigmoid(output)
                output = output.reshape(target.shape)
                target = target.type(torch.float)
                
                # Compute Loss
                loss = self.criterion(output, target)

                # Record Loss (Updates Average Meter With The Accuracy Value)
                losses.update(loss.item(), input.size(0))

                if (self.optim != 'AUC'):
                    output = torch.sigmoid(output)

                # Add To Total Outputs & Targets
                outputs.append(output.detach().cpu())
                targets.append(target.detach().cpu())

                if (self.verbose):
                    pbar.update(input.size(0))

            if (self.verbose):
                pbar.close()
    
        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)

        probs = outputs
        threshold = utils.get_threshold(targets, probs)
        
        _, _, AUC = utils.get_ROC_AUC(targets, probs)
        preds = (probs >= threshold).astype(int).reshape(-1)
        acc1, precision_scores, recall_scores, f1_scores = utils.get_metrics(targets, preds)

        if (self.verbose):
            print('[ --- Validation --- ]')
            print('Loss      : %.4f'%(losses.avg))
            print('Top 1 Acc : %.4f'%(acc1))
            print('ROC-AUC   : %.4f'%(AUC))
            print ('---------------------') 
            print('Precision [ Negative : %.4f | Positive : %.4f | Average : %.4f ]'%(precision_scores[0], precision_scores[1], np.mean(precision_scores)))
            print('Recall    [ Negative : %.4f | Positive : %.4f | Average : %.4f ]'%(recall_scores[0], recall_scores[1], np.mean(recall_scores)))
            print('F1        [ Negative : %.4f | Positive : %.4f | Average : %.4f ]'%(f1_scores[0], f1_scores[1], np.mean(f1_scores)))
            
        return losses.avg, acc1, AUC, np.mean(f1_scores), threshold

    # Evaludate
    def evaluate(self, dataloader, model_pth, save_dir):

        # Load Model
        checkpoint = torch.load(model_pth)
        self.model.load_state_dict(checkpoint['state_dict'])
        threshold = checkpoint['threshold']

        outputs = []
        targets = []

        # switch to evaluate mode
        self.model.eval()

        if (self.verbose):
            pbar = tqdm(desc='Test Loop', total=len(dataloader.dataset), dynamic_ncols=True)

        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(dataloader):
                
                input = input.cuda(self.device)
                target = target.cuda(self.device)

                # Compute Output
                output = self.forward(input)
                output = torch.sigmoid(output)

                # Add To Total Outputs & Targets
                outputs.append(output.detach().cpu())
                targets.append(target.detach().cpu())

                if (self.verbose):
                    pbar.update(input.size(0))
            
            if (self.verbose):
                pbar.close()
    
        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)

        if (self.n_output == 1):
            probs = outputs
        else:
            probs = outputs[:, 1]

        preds = (probs >= threshold).astype(int).reshape(-1)

        title = 'Confusion Matrix | Threshold [%.4f]'%(threshold)
        utils.save_confmat(targets, preds, title, save_dir)

        acc1, precision_scores, recall_scores, f1_scores = utils.save_metrics(targets, preds, save_dir)

        AUC = utils.save_ROC_curve(targets, probs, save_dir)

        labels = {
            'xlabel' : 'Output Probability',
            'ylabel' : 'Density',
            'title' : 'Probability Histogram',
        }
        utils.save_histogram(probs, labels, os.path.join(save_dir, 'histogram.png'))

        if (self.verbose):
            print('Model Name : %s'%(model_pth))
        
        return acc1, precision_scores, recall_scores, f1_scores, AUC
        
class ConvNeXtB_Module(Base_Module):
    def __init__(self, args, verbose, imratio, train_iters_per_epoch=None):
        Base_Module.__init__(self, args, verbose)

        if (verbose):
            print('\nLoading ConvNeXt-B Transformer Model ...')

        if (self.pretrained == 'SimCLR'):
            raise NotImplementedError
        
        elif (self.pretrained == 'MoCo'):
            raise NotImplementedError

        elif (self.pretrained == 'Supervised'):
            self.model = timm.create_model('convnext_base', pretrained=True, num_classes=self.n_output)
            if (verbose):
                print('Loaded Supervised ConvNeXt-B Model on ImageNet')

        elif (self.pretrained == 'None'):
            self.model = timm.create_model('convnext_base', pretrained=False, num_classes=self.n_output)
            if (verbose):
                print('Loaded ConvNeXt-B Model with random weights')
            
        else:
            raise NotImplementedError
            
        self.model = self.model.cuda(self.device)
        
        self.construct_optimizer(imratio)
        
    def forward(self, x):
        out = self.model(x)
        return out

class SwinB_Module(Base_Module):
    def __init__(self, args, verbose, imratio, train_iters_per_epoch=None):
        Base_Module.__init__(self, args, verbose)

        if (verbose):
            print('\nLoading Swin-B Transformer Model ...')

        if (self.pretrained == 'SimCLR'):
            raise NotImplementedError
        
        elif (self.pretrained == 'MoCo'):
            raise NotImplementedError

        elif (self.pretrained == 'Supervised'):
            self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=self.n_output, img_size=224)
            if (verbose):
                print('Loaded Supervised Swin-B Model on ImageNet')

        elif (self.pretrained == 'None'):
            self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=self.n_output, img_size=224)
            if (verbose):
                print('Loaded Swin-B Model with random weights')
            
        else:
            raise NotImplementedError
            
        self.model = self.model.cuda(self.device)
        
        self.construct_optimizer(imratio)
        
    def forward(self, x):
        out = self.model(x)
        return out
        
class SwinT_Module(Base_Module):
    def __init__(self, args, verbose, imratio, train_iters_per_epoch=None):
        Base_Module.__init__(self, args, verbose)

        if (verbose):
            print('\nLoading Swin-T Transformer Model ...')

        if (self.pretrained == 'SimCLR'):
            raise NotImplementedError
        
        elif (self.pretrained == 'MoCo'):
            raise NotImplementedError

        elif (self.pretrained == 'Supervised'):
            self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=self.n_output, img_size=224)
            if (verbose):
                print('Loaded Supervised Swin-T Model on ImageNet')

        elif (self.pretrained == 'None'):
            self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=self.n_output, img_size=224)
            if (verbose):
                print('Loaded Swin-T Model with random weights')
            
        else:
            raise NotImplementedError
            
        self.model = self.model.cuda(self.device)
        
        self.construct_optimizer(imratio)
        
    def forward(self, x):
        out = self.model(x)
        return out

class SwinS_Module(Base_Module):
    def __init__(self, args, verbose, imratio, train_iters_per_epoch=None):
        Base_Module.__init__(self, args, verbose)

        if (verbose):
            print('\nLoading Swin-S Transformer Model ...')

        if (self.pretrained == 'SimCLR'):
            raise NotImplementedError
        
        elif (self.pretrained == 'MoCo'):
            raise NotImplementedError

        elif (self.pretrained == 'Supervised'):
            self.model = timm.create_model('swin_small_patch4_window7_224', pretrained=True, num_classes=self.n_output, img_size=224)
            if (verbose):
                print('Loaded Supervised Swin-S Model on ImageNet')

        elif (self.pretrained == 'None'):
            self.model = timm.create_model('swin_small_patch4_window7_224', pretrained=False, num_classes=self.n_output, img_size=224)
            if (verbose):
                print('Loaded Swin-S Model with random weights')
            
        else:
            raise NotImplementedError
            
        self.model = self.model.cuda(self.device)
        
        self.construct_optimizer(imratio)
        
    def forward(self, x):
        out = self.model(x)
        return out
        
class SwinL_Module(Base_Module):
    def __init__(self, args, verbose, imratio, train_iters_per_epoch=None):
        Base_Module.__init__(self, args, verbose)

        if (verbose):
            print('\nLoading Swin-S Transformer Model ...')

        if (self.pretrained == 'SimCLR'):
            raise NotImplementedError
        
        elif (self.pretrained == 'MoCo'):
            raise NotImplementedError

        elif (self.pretrained == 'Supervised'):
            self.model = timm.create_model('swin_small_patch4_window7_224', pretrained=True, num_classes=self.n_output, img_size=224)
            if (verbose):
                print('Loaded Supervised Swin-L Model on ImageNet')

        elif (self.pretrained == 'None'):
            self.model = timm.create_model('swin_small_patch4_window7_224', pretrained=False, num_classes=self.n_output, img_size=224)
            if (verbose):
                print('Loaded Swin-L Model with random weights')
            
        else:
            raise NotImplementedError
            
        self.model = self.model.cuda(self.device)
        
        self.construct_optimizer(imratio)
        
    def forward(self, x):
        out = self.model(x)
        return out
        
class Resnet50_Module(Base_Module):
    def __init__(self, args, verbose, imratio, train_iters_per_epoch=None):
        Base_Module.__init__(self, args, verbose)

        if (verbose):
            print('\nLoading Resnet50 Model ...')

        if (self.pretrained == 'SimCLR'):
            self.model, _ = get_resnet(depth=50, width_multiplier=1, sk_ratio=0)
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, (self.model.fc.in_features // 2)),
                nn.ReLU(),
                nn.Linear((self.model.fc.in_features // 2), self.n_output)
            )
            assert(os.path.exists(args.ckpt_path))

            state_dict = torch.load(args.ckpt_path)['encoder']
            msg = self.model.load_state_dict(state_dict, strict=False)
            assert(msg.missing_keys == ['fc.0.weight', 'fc.0.bias', 'fc.2.weight', 'fc.2.bias'])
            if (verbose):
                print('Loaded SimCLR Model From %s'%(args.ckpt_path))
        
        elif (self.pretrained == 'MoCo'):
            self.model = models.__dict__['resnet50']()
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, (self.model.fc.in_features // 2)),
                nn.ReLU(),
                nn.Linear((self.model.fc.in_features // 2), self.n_output)
            )
            assert(os.path.exists(args.ckpt_path))

            state_dict = torch.load(args.ckpt_path)['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if (k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.fc')):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            msg = self.model.load_state_dict(state_dict, strict=False)
            assert(msg.missing_keys == ['fc.0.weight', 'fc.0.bias', 'fc.2.weight', 'fc.2.bias'])
            if (verbose):
                print('Loaded MoCo Model From %s'%(args.ckpt_path))

        elif (self.pretrained == 'Supervised'):
            self.model = models.__dict__['resnet50'](pretrained=True)
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, (self.model.fc.in_features // 2)),
                nn.ReLU(),
                nn.Linear((self.model.fc.in_features // 2), self.n_output)
            )
            if (verbose):
                print('Loaded Supervised Tensorflow Model on ImageNet')

        elif (self.pretrained == 'None'):
            self.model = models.__dict__['resnet50'](pretrained=False)
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, (self.model.fc.in_features // 2)),
                nn.ReLU(),
                nn.Linear((self.model.fc.in_features // 2), self.n_output)
            )
            if (verbose):
                print('\nLoaded Model With Random Weights')

        else:
            raise NotImplementedError

        self.model = self.model.cuda(self.device)

        if (self.dist):

            # Requires Testing
            #self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            
            self.model = DDP(self.model, device_ids=[self.device])

        self.construct_optimizer(imratio)

    def forward(self, x):
        if (self.pretrained == 'SimCLR'):
            out = self.model(x, apply_fc=True)
        else:
            out = self.model(x)
        return out

class Densenet121_Module(Base_Module):
    def __init__(self, args, verbose, imratio, train_iters_per_epoch=None):
        Base_Module.__init__(self, args, verbose)

        if (verbose):
            print('\nLoading Densenet121 Model')

        if (self.pretrained == 'SimCLR'):
            raise NotImplementedError
        elif (self.pretrained == 'MoCo'):
            assert(os.path.exists(args.ckpt_path))
            pretrained_dict = torch.load(args.ckpt_path)["state_dict"]
            state_dict = {}
            for k, v in pretrained_dict.items():
                if k.startswith("model.encoder_q."):
                    k = k.replace("model.encoder_q.", "")
                    state_dict[k] = v

            # Delete Classifier Weights
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']

            self.model = models.__dict__['densenet121'](num_classes=self.n_output)
            msg = self.model.load_state_dict(state_dict, strict=False)
            assert(msg.missing_keys == ['classifier.weight', 'classifier.bias'])

            if (verbose):
                print('\nLoaded MoCo Model From %s'%(args.ckpt_path))

        elif (self.pretrained == 'Supervised'):
            self.model = models.__dict__['densenet121'](pretrained=True)
            self.model.classifier = nn.Linear(in_features=self.model.classifier.in_features, out_features=self.n_output, bias=True)
    
            if (verbose):
                print('Loaded Supervised Tensorflow Model on ImageNet')

        elif (self.pretrained == 'None'):
            self.model = models.__dict__['densenet121'](num_classes=self.n_output)

            if (verbose):
                print('\nLoaded Model With Random Weights')

        else:
            raise NotImplementedError

        self.model = self.model.cuda(self.device)

        if (self.dist):

            # Requires Testing
            #self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            
            self.model = DDP(self.model, device_ids=[self.device])

        self.construct_optimizer(imratio)
    
    def forward(self, x):
        out = self.model(x)
        return out

################################################ Pretraining Modules #############################################

class SyncFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]

class NT_Xent(nn.Module):
    def __init__(self, temperature=0.5, eps=1e-6):
        super(NT_Xent, self).__init__()

        self.temperature = temperature
        self.eps = eps

    def forward(self, z_i, z_j):
        
        """
        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(z_i)
            out_2_dist = SyncFunction.apply(z_j)
        else:
            out_1_dist = z_i
            out_2_dist = z_j

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([z_i, z_j], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / self.temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = Tensor(neg.shape).fill_(math.e ** (1 / self.temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=self.eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(z_i * z_j, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + self.eps)).mean()

        return loss

class SimCLR(nn.Module):
    def __init__(self, normalize_projector_output=True):
        super().__init__()

        self.normalize = normalize_projector_output

        self.encoder, self.projector = get_resnet(depth=50, width_multiplier=1, sk_ratio=0)

    def forward(self, x_i, x_j):

        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        if (self.normalize):
            z_i = F.normalize(z_i, dim=-1)
            z_j = F.normalize(z_j, dim=-1)

        return z_i, z_j

class SimCLR_Module():
    def __init__(self, args, verbose):

        self.n_epochs = args.n_epochs
        self.dist = args.dist
        self.device = args.device
        self.optim = args.optim
        self.verbose = verbose

        self.simclr = SimCLR()
        self.simclr = self.simclr.cuda(self.device)

        if (self.dist):

            # Requires Testing
            #self.simclr = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.simclr)
            
            self.simclr = DDP(self.simclr, device_ids=[self.device])

        self.criterion = NT_Xent()
        
        if (self.optim == 'SGD'):
            
            self.optimizer = torch.optim.SGD(
                self.simclr.parameters(),
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.weight_decay,
            )
        
        elif (self.optim == 'Adam'):

            raise NotImplementedError

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_epochs)

    def get_lr(self):
        return self.scheduler.get_last_lr()[-1]

    def train(self, dataloader, epoch):

        losses = utils.AverageMeter()

        # Switch To Train Mode
        self.simclr.train()

        _len = len(dataloader) * dataloader.batch_size if (self.dist) else len(dataloader.dataset) 

        if (self.verbose):
            pbar = tqdm(desc='Train Loop', total=_len , dynamic_ncols=True)

        for input_i, input_j in dataloader:

            input_i = input_i.cuda(self.device)
            input_j = input_j.cuda(self.device) 

            z_i, z_j = self.simclr(input_i, input_j)

            loss = self.criterion(z_i, z_j)
            
            losses.update(loss.item(), input_i.size(0))

            # Compute Gradient And Do Optimizer Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (self.verbose):
                pbar.update(input_i.size(0))

        # Update Learning Rate
        self.scheduler.step()

        if (self.verbose):
            pbar.close()    
            print('\nEpoch [ %d / %d ]'%(epoch, self.n_epochs))

        return losses.avg
