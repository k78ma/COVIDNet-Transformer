from argparse import ArgumentParser

###########################################################################################################################
###                                                Args For Finetuning                                                  ###
###########################################################################################################################
def add_finetune_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    
    parser.add_argument('--seed', default=123, 
        type=int, help='seed for dataset generation')

    parser.add_argument('--arch', default='convnext-b',
        type=str, choices=['resnet50', 'densenet121', 'swin-b', 'swin-t', 'swin-s', 'swin-l', 'convnext-b'], help='model architecture')
    parser.add_argument('--ckpt-path', default='resnet_models/simclr_resnet50.pth', 
        type=str, help='path to pretrained model checkpoint')
    parser.add_argument('--pretrained', default='Supervised',
        type=str, choices=['SimCLR', 'MoCo', 'Supervised', 'None'], help='which pretrained checkpoint to use')
    parser.add_argument('--data-dir', default="../archive", 
        type=str, help='path to dataset')
    parser.add_argument('--exp-dir', default="./test100_convnext_CE", 
        type=str, help='export directory')

    parser.add_argument('--n-epochs', default=30, 
        type=int, help='number of total epochs to run')
    parser.add_argument('--batch-size', default=64, 
        type=int, help='mini-batch size')
    parser.add_argument('--uniform', 
        action='store_true', help='enable uniform sampling')

    parser.add_argument('--optim', default='SGD',
        type=str, choices=['SGD', 'AUC'], help='which optimizer to use')
    parser.add_argument('--lr', default=3e-4, 
        type=float, help='initial learning rate') 
    parser.add_argument('--weight-decay', default=1e-4, 
        type=float, help='weight decay for embedding and classifier parameters') # Only Used For SGD Optimization
    '''
    parser.add_argument('--warmup-epochs', default=10, 
        type=float, help='number of linear warm-up epochs')
    '''
    parser.add_argument('--gamma', default=500,
        type=int, help='gamma for AUC maximization') # Only Used For AUC Maximization
    parser.add_argument('--margin', default=1.0,
        type=float, help='margin for AUC maximization') # Only Used For AUC Maximization

    parser.add_argument('--dist', 
        action='store_true', help='enable for distributed training')
    parser.add_argument('--num-nodes', default=1, 
        type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, 
        type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:1234', 
        type=str, help='url used for distributed training')

    return parser

###########################################################################################################################
###                                                Args For K-Fold Finetuning                                           ###
###########################################################################################################################
def add_kfold_finetune_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    
    parser.add_argument('--seed', default=123, 
        type=int, help='seed for dataset generation')

    parser.add_argument('--n_folds', default=5, 
        type=int, help='number of folds for cross-validation')

    parser.add_argument('--arch', default='resnet50',
        type=str, choices=['resnet50', 'densenet121'], help='model architecture')
    parser.add_argument('--ckpt-path', default='resnet_models/simclr_resnet50.pth', 
        type=str, help='path to pretrained model checkpoint')
    parser.add_argument('--pretrained', default='SimCLR',
        type=str, choices=['SimCLR', 'MoCo', 'Supervised', 'None'], help='which pretrained checkpoint to use')
    parser.add_argument('--data-dir', default="../archive", 
        type=str, help='path to dataset')
    parser.add_argument('--exp-dir', default="./test50_kfold", 
        type=str, help='export directory')

    parser.add_argument('--n-epochs', default=50, 
        type=int, help='number of total epochs to run')
    parser.add_argument('--batch-size', default=128, 
        type=int, help='mini-batch size')
    parser.add_argument('--uniform', 
        action='store_true', help='enable uniform sampling')

    parser.add_argument('--optim', default='SGD',
        type=str, choices=['SGD', 'AUC'], help='which optimizer to use')
    parser.add_argument('--lr', default=1e-3, 
        type=float, help='initial learning rate') # 1e-3 for SGD Optimization, 0.1 for AUC
    parser.add_argument('--weight-decay', default=1e-4, 
        type=float, help='weight decay for embedding and classifier parameters') # Only Used For SGD Optimization
    '''
    parser.add_argument('--warmup-epochs', default=0, 
        type=float, help='number of linear warm-up epochs')
    '''
    parser.add_argument('--gamma', default=500,
        type=int, help='gamma for AUC maximization') # Only Used For AUC Maximization
    parser.add_argument('--margin', default=1.0,
        type=float, help='margin for AUC maximization') # Only Used For AUC Maximization

    parser.add_argument('--dist', 
        action='store_true', help='enable for distributed training')
    parser.add_argument('--num-nodes', default=1, 
        type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, 
        type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:1234', 
        type=str, help='url used for distributed training')

    return parser

###########################################################################################################################
###                                                Args For Pretraining                                                 ###
###########################################################################################################################

def add_pretrain_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    
    parser.add_argument('--seed', default=123, 
        type=int, help='seed for dataset generation')

    parser.add_argument('--ckpt-path', default='densenet_models/moco_densenet121.pt', 
        type=str, help='path to pretrained model checkpoint')
    parser.add_argument('--data-dir', default="../CheXpert-v1.0-small", 
        type=str, help='path to dataset')
    parser.add_argument('--exp-dir', default="./test", 
        type=str, help='export directory')

    parser.add_argument('--n-epochs', default=100, 
        type=int, help='number of total epochs to run')
    parser.add_argument('--batch-size', default=128, 
        type=int, help='mini-batch size')

    parser.add_argument('--optim', default='SGD',
        type=str, choices=['SGD', 'Adam'], help='which optimizer to use')
    parser.add_argument('--lr', default=0.1, 
        type=float, help='initial learning rate') 
    parser.add_argument('--weight-decay', default=1e-4, 
        type=float, help='weight decay for embedding and classifier parameters') 

    parser.add_argument('--dist', 
        action='store_true', help='enable for distributed training')
    parser.add_argument('--num-nodes', default=1, 
        type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, 
        type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:1234', 
        type=str, help='url used for distributed training')

    return parser

###########################################################################################################################
###                                                Args For GradCAM Testing                                             ###
###########################################################################################################################

def add_gradCam_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    
    parser.add_argument('--n_output', default=1,
        type=int, choices=[1, 2], help='how many output neurons to use')

    parser.add_argument('--seed', default=123, 
        type=int, help='seed for dataset generation')

    parser.add_argument('--model_pth', default="./30epoch_models/sgd_mimic.ckpt", 
        type=str, help='path to pretrained simCLR model checkpoint')
    parser.add_argument('--data_dir', default="../archive", 
        type=str, help='path to dataset')
    parser.add_argument('--exp_dir', default="./SGD_Mimic", 
        type=str, help='export directory')

    return parser
