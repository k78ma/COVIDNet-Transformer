import os
import numpy as np
from PIL import Image
import pandas as pd

from torchvision import transforms
import torchvision.transforms.functional as F

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from samplers import DistributedSamplerWrapper
from sklearn.model_selection import StratifiedKFold

class_map = {
    'negative' : 0,
    'positive' : 1,
}

def _process_txt_file(file):
    with open(file, 'r') as fr:
        files = fr.readlines()
        files = [x.split()[-3:-1] for x in files]

    return files

def _process_csv_file(file):
    data = pd.read_csv(file)
    data = data.loc[data['Frontal/Lateral'] == 'Frontal'] # Keep only frontal x-rays
    data = data['Path']

    return list(data)

################################################## Transformations ############################################################

# Normalization Used On The Image, ImageNet Normalization
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

class Transform_Finetuning:
    def __init__(self, img_size=224):

        self.transform_train = transforms.Compose([
            #transforms.RandomRotation(degrees=(-20, 20)),
            SquarePad(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size), # Cropped Because Sometimes Resize Doesnt Perfectly Produce 224x224
            transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
         ])

        self.transform_valid = transforms.Compose([
            SquarePad(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size), # Cropped Because Sometimes Resize Doesnt Perfectly Produce 224x224
            transforms.ToTensor(),
            normalize,
        ])

        self.transform_cam = transforms.Compose([
            SquarePad(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size), # Cropped Because Sometimes Resize Doesnt Perfectly Produce 224x224
            transforms.ToTensor(),
        ])

    def __call__(self, img, mode='train'):
        
        if (mode == 'train'):
            return self.transform_train(img)
        elif (mode == 'valid'):
            return self.transform_valid(img)
        elif (mode == 'cam'):
            return self.transform_cam(img)

class Transform_Pretraining:
    """
    Applies different transforms to one image to produce two (x_i, x_j) outputs. Transforms here
    are as per the recommended in the simCLR paper: https://arxiv.org/pdf/2002.05709.pdf
    """
    def __init__(self, img_size=224):
        
        # Params Based Off Of Official SimCLR TF Implementation: https://github.com/google-research/simclr, But Adjusted To Fit Better With CovidX Dataset
        s = 0.5
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )

        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-45, 45)),
            transforms.RandomCrop(size=img_size),
            
            # With 0.5 Probability By Default 
            transforms.RandomHorizontalFlip(),  
            transforms.RandomApply([color_jitter], p=0.5),
            #transforms.RandomGrayscale(p=0.5),
            
            # Paper States Gaussian Blur Used During Training, But In Github Its Not, Omitting For Now.
            # transforms.GaussianBlur(kernel_size=(1,7), sigma(0.1, 2)), 
            transforms.ToTensor()
         ])

    def __call__(self, img):
        return self.transform(img), self.transform(img)

################################################## Finetuning Data Handlers ###################################################

class Dataset_CovidX_Finetuning(Dataset):
    def __init__(self, img_paths, img_labels, is_train=True):
        super().__init__()

        self.img_paths = img_paths
        self.img_labels = img_labels
        self.is_train = is_train

        # Calculate Imbalance Ratio (For AUC)
        self.imratio = {L : len(np.where(self.img_labels == L)[0]) / self.img_labels.shape[0] for L in np.unique(self.img_labels)}
        
        # Calculate Weights For Uniform Sampling
        n_class_samples = np.array([len(np.where(self.img_labels == L)[0]) for L in np.unique(self.img_labels)])
        class_weights = 1.0 / n_class_samples
        self.sample_weights = np.array([class_weights[L] for L in self.img_labels])
        self.sample_weights = torch.from_numpy(self.sample_weights).float()

        self.transform = Transform_Finetuning()

    def __len__(self):
        return self.img_paths.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if (self.is_train):
            img = self.transform(image, mode='train')
        else:
            img = self.transform(image, mode='valid')

        label = int(self.img_labels[idx])
        
        return img, label

class DataHandler_CovidX_Finetuning(): 
    def __init__(self, data_dir, val_split=0.1, verbose=True):
        
        self.data_dir = data_dir
        self.val_split = val_split 
        self.verbose = verbose
        self.im_count = {}

        if (verbose):
            print('\nClassifying Negative/Positive')

        # Read CovidX Train And Test Files
        train_files = _process_txt_file(os.path.join(self.data_dir, "train_9B.txt"))
        test_files = _process_txt_file(os.path.join(self.data_dir, "test_9B.txt"))

        train_files = [[os.path.join(self.data_dir, 'train', pth), label] for pth, label in train_files]
        test_files = [[os.path.join(self.data_dir, 'test', pth), label] for pth, label in test_files]

        train_files = np.asarray(train_files)
        test_files = np.asarray(test_files)

        # create validation split
        val_files = None
        if self.val_split > 0.0:
            order = np.random.permutation(train_files.shape[0])
            cut_off = int(train_files.shape[0] * (1.0 - self.val_split))
            
            val_files = train_files[order[cut_off:]]
            train_files = train_files[order[:cut_off]]

        # Seperate Data Into Train/Test/Valid
        self.train_img_paths, self.train_labels = self.seperate_data(
            task='train',
            files=train_files
        )

        self.test_img_paths, self.test_labels = self.seperate_data(
            task='test',
            files=test_files
        )

        self.val_img_paths = None
        self.val_labels = None
        if (val_files is not None):
            self.val_img_paths, self.val_labels = self.seperate_data(
                task='valid',
                files=val_files
            )

    def seperate_data(self, task, files):
        
        self.im_count[task] = {
            'negative' : 0,
            'positive' : 0
        }

        img_paths = []
        labels = []

        for fname, label in files:
            img_paths.append(fname)

            self.im_count[task][label] += 1
            labels.append(class_map[label])

        if (self.verbose):
            print("\nnumber of negative cases in %s split: "%(task), self.im_count[task]['negative'])
            print("number of positive cases in %s split: "%(task), self.im_count[task]['positive'])
            
        return np.asarray(img_paths), np.asarray(labels)

    def get_datasets(self):

        train_dataset = Dataset_CovidX_Finetuning(self.train_img_paths, self.train_labels, is_train=True)
        test_dataset = Dataset_CovidX_Finetuning(self.test_img_paths, self.test_labels, is_train=False)
        val_dataset = None
        if ((self.val_img_paths is not None) and (self.val_labels is not None)):
            val_dataset = Dataset_CovidX_Finetuning(self.val_img_paths, self.val_labels, is_train=False)

        return train_dataset, test_dataset, val_dataset
    
    def get_dataloaders(self, args):
    
        train_dataset, test_dataset, val_dataset = self.get_datasets()
        
        sampler = None
        if (args.uniform):
            
            # Uniform Sampling
            sampler = WeightedRandomSampler(
                train_dataset.sample_weights, 
                len(train_dataset.sample_weights), 
                replacement=True
            )

            if (self.verbose):
                print('\nEnabling Uniform Sampling')
        else:

            if (self.verbose):
                print('\nDisabling Uniform Sampling')

        if (args.dist):
            sampler = DistributedSampler(train_dataset) if (sampler is None) else DistributedSamplerWrapper(sampler)

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            sampler=sampler,
            shuffle=(sampler is None),
            drop_last=True,
        )

        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            sampler=None,
            shuffle=False
        )
        
        val_dataloader = None
        if (val_dataset is not None):
            val_dataloader = DataLoader(
                val_dataset, 
                batch_size=args.batch_size, 
                sampler=None,
                shuffle=False
            )
        
        return train_dataloader, test_dataloader, val_dataloader, sampler

class DataHandler_KFold_CovidX_Finetuning():
    def __init__(self, data_dir, n_folds=5, verbose=True):
        
        self.data_dir = data_dir
        self.verbose = verbose
        self.im_count = {}

        if (verbose):
            print('\nClassifying Negative/Positive')

        # Read CovidX Train And Test Files
        train_files = _process_txt_file(os.path.join(self.data_dir, "train_8B.txt"))
        test_files = _process_txt_file(os.path.join(self.data_dir, "test_8B.txt"))

        train_files = [[os.path.join(self.data_dir, 'train', pth), label] for pth, label in train_files]
        test_files = [[os.path.join(self.data_dir, 'test', pth), label] for pth, label in test_files]

        self.train_img_paths, self.train_labels = self.seperate_data(
            task='train',
            files=train_files
        )

        self.test_img_paths, self.test_labels = self.seperate_data(
            task='test',
            files=test_files
        )

        # Shuffle Image Paths
        random_order = np.random.permutation(self.train_labels.shape[0])
        self.train_img_paths = self.train_img_paths[random_order]
        self.train_labels = self.train_labels[random_order]
        
        self.skf = StratifiedKFold(n_splits=n_folds)
    
    def generate_folds(self, args):
        
        for fold, (train_indices, valid_indices) in enumerate(self.skf.split(self.train_img_paths, self.train_labels)):

            fold_train_img_paths = self.train_img_paths[train_indices]
            fold_train_labels = self.train_labels[train_indices]
            
            fold_val_img_paths = self.train_img_paths[valid_indices]
            fold_val_labels = self.train_labels[valid_indices]

            if (self.verbose):
                print('\nFold %d'%(fold + 1))
                print('-' * 100)

                n_zeros = sum(fold_train_labels == 0)
                n_ones = sum(fold_train_labels == 1)
                print('Train Data [ Negative : %d | Positive : %d | Total : %d ]'%(n_zeros, n_ones, len(train_indices)))

                n_zeros = sum(fold_val_labels == 0)
                n_ones = sum(fold_val_labels == 1)
                print('Valid Data [ Negative : %d | Positive : %d | Total : %d ]'%(n_zeros, n_ones, len(valid_indices)))

            train_dataset = Dataset_CovidX_Finetuning(fold_train_img_paths, fold_train_labels, is_train=True)
            val_dataset = Dataset_CovidX_Finetuning(fold_val_img_paths, fold_val_labels, is_train=False)

            sampler = None
            if (args.uniform):
                
                # Uniform Sampling
                sampler = WeightedRandomSampler(
                    train_dataset.sample_weights, 
                    len(train_dataset.sample_weights), 
                    replacement=True
                )

                if (self.verbose):
                    print('\nEnabling Uniform Sampling')
            else:

                if (self.verbose):
                    print('\nDisabling Uniform Sampling')

            if (args.dist):
                sampler = DistributedSampler(train_dataset) if (sampler is None) else DistributedSamplerWrapper(sampler)

            train_dataloader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                sampler=sampler,
                shuffle=(sampler is None),
                drop_last=True,
            )

            val_dataloader = DataLoader(
                val_dataset, 
                batch_size=args.batch_size, 
                sampler=None,
                shuffle=False
            )
            
            yield fold + 1, train_dataloader, val_dataloader, sampler

    def seperate_data(self, task, files):
        
        self.im_count[task] = {
            'negative' : 0,
            'positive' : 0
        }

        img_paths = []
        labels = []

        for fname, label in files:
            img_paths.append(fname)

            self.im_count[task][label] += 1
            labels.append(class_map[label])

        if (self.verbose):
            print("\nnumber of negative cases in %s split: "%(task), self.im_count[task]['negative'])
            print("number of positive cases in %s split: "%(task), self.im_count[task]['positive'])
            
        return np.asarray(img_paths), np.asarray(labels)

    def get_test_dataloader(self, args):
        
        test_dataset = Dataset_CovidX_Finetuning(self.test_img_paths, self.test_labels, is_train=False)
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            sampler=None,
            shuffle=False
        )
    
        return test_dataloader

################################################## Pretraining Data Handlers ##################################################

class Dataset_CheXpert_Pretraining(Dataset):
    def __init__(self, img_paths):
        super().__init__()
        
        self.img_paths = np.asarray(img_paths)

        self.transform = Transform_Pretraining()

    def __len__(self):
        return self.img_paths.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        img1, img2 = self.transform(image)
        return img1, img2 

class DataHandler_CheXpert_Pretraining():
    def __init__(self, data_dir):
        
        self.data_dir = data_dir

        # Read Train/Test CSV Split
        train_files = _process_csv_file(os.path.join(data_dir, 'train.csv'))
        test_files = _process_csv_file(os.path.join(data_dir, 'valid.csv'))

        img_paths = []

        for fname in train_files:
            img_paths.append(os.path.join("../", fname))
        for fname in test_files:
            img_paths.append(os.path.join("../", fname))

        self.img_paths = np.asarray(img_paths)

    def get_datasets(self):
        
        dataset = Dataset_CheXpert_Pretraining(self.img_paths)
        return dataset

    def get_dataloaders(self, args):
        
        dataset = self.get_datasets()

        sampler = None
        if (args.dist):
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler)

        return dataloader, sampler

################################################## GradCAM Data Handlers ######################################################

class Dataset_CovidX_GradCAM(Dataset):
    def __init__(self, img_paths, img_labels):
        super().__init__()

        self.img_paths = img_paths
        self.img_labels = img_labels

        # Calculate Imbalance Ratio (For AUC)
        self.imratio = {L : len(np.where(self.img_labels == L)[0]) / self.img_labels.shape[0] for L in np.unique(self.img_labels)}
        
        # Calculate Weights For Uniform Sampling
        n_class_samples = np.array([len(np.where(self.img_labels == L)[0]) for L in np.unique(self.img_labels)])
        class_weights = 1.0 / n_class_samples
        self.sample_weights = np.array([class_weights[L] for L in self.img_labels])
        self.sample_weights = torch.from_numpy(self.sample_weights).float()

        self.transform = Transform_Finetuning()

    def __len__(self):
        return self.img_paths.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = os.path.split(img_path)[-1]
        
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image, mode='cam')
        input = self.transform(image, mode='valid')

        label = int(self.img_labels[idx])
        
        return img_name, img, input, label

class DataHandler_CovidX_GradCAM():
    def __init__(self, data_dir, val_split=0.1, verbose=True):
        
        self.data_dir = data_dir
        self.val_split = val_split 
        self.verbose = verbose
        self.im_count = {}


        if (verbose):
            print('\nClassifying Negative/Positive')

        # Read CovidX Train And Test Files
        train_files = _process_txt_file(os.path.join(self.data_dir, "train_8B.txt"))
        test_files = _process_txt_file(os.path.join(self.data_dir, "test_8B.txt"))

        train_files = [[os.path.join(self.data_dir, 'train', pth), label] for pth, label in train_files]
        test_files = [[os.path.join(self.data_dir, 'test', pth), label] for pth, label in test_files]

        train_files = np.asarray(train_files)
        test_files = np.asarray(test_files)

        # create validation split
        val_files = None
        if self.val_split > 0.0:
            order = np.random.permutation(train_files.shape[0])
            cut_off = int(train_files.shape[0] * (1.0 - self.val_split))
            
            val_files = train_files[order[cut_off:]]
            train_files = train_files[order[:cut_off]]

        # Seperate Data Into Train/Test/Valid
        self.train_img_paths, self.train_labels = self.seperate_data(
            task='train',
            files=train_files
        )

        self.test_img_paths, self.test_labels = self.seperate_data(
            task='test',
            files=test_files
        )

        self.val_img_paths = None
        self.val_labels = None
        if (val_files is not None):
            self.val_img_paths, self.val_labels = self.seperate_data(
                task='valid',
                files=val_files
            )

    def seperate_data(self, task, files):
        
        self.im_count[task] = {
            'negative' : 0,
            'positive' : 0
        }

        img_paths = []
        labels = []

        for fname, label in files:
            img_paths.append(fname)

            self.im_count[task][label] += 1
            labels.append(class_map[label])

        if (self.verbose):
            print("\nnumber of negative cases in %s split: "%(task), self.im_count[task]['negative'])
            print("number of positive cases in %s split: "%(task), self.im_count[task]['positive'])
            
        return np.asarray(img_paths), np.asarray(labels)

    def get_datasets(self):

        train_dataset = Dataset_CovidX_GradCAM(self.train_img_paths, self.train_labels)
        test_dataset = Dataset_CovidX_GradCAM(self.test_img_paths, self.test_labels)
        val_dataset = None
        if ((self.val_img_paths is not None) and (self.val_labels is not None)):
            val_dataset = Dataset_CovidX_GradCAM(self.val_img_paths, self.val_labels)

        return train_dataset, test_dataset, val_dataset

    def get_dataloaders(self):
    
        train_dataset, test_dataset, val_dataset = self.get_datasets()
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=1, 
            sampler=None,
            shuffle=False
        )

        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=1, 
            sampler=None,
            shuffle=False
        )
        
        val_dataloader = None
        if (val_dataset is not None):
            val_dataloader = DataLoader(
                val_dataset, 
                batch_size=1, 
                sampler=None,
                shuffle=False
            )
        
        return train_dataloader, test_dataloader, val_dataloader
