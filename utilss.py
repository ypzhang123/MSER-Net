import torch
import os
import numpy as np
from Attack import ImageAttacks
import cv2
import random
from torch.utils import data
from torchvision import transforms as trans
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as cal_auc
from PIL import Image
import sys
import logging


#trans.RandomHorizontalFlip(p=0.5),
#trans.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), 
class FFDataset(data.Dataset):
    def __init__(self, dataset_root, frame_num=32, size=299, augment=True, apply_attack=False, attack_severity=1, attack_method=None):
        self.data_root = dataset_root
        self.frame_num = frame_num
        self.train_list = self.collect_image(self.data_root)
        self.size = size
        self.apply_attack = apply_attack
        self.attack_severity = attack_severity
        self.attack_method = attack_method

        if augment:
            self.transform = trans.Compose([
                trans.RandomHorizontalFlip(p=0.5), trans.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), trans.ToTensor()
            ])
        else:
            self.transform = trans.ToTensor()

        self.max_val = 1.
        self.min_val = -1.

        # ���ù��������б�������֤
        self.valid_attack_methods = [
            'color_saturation',
            'color_contrast',
            'blockwise',
            'gaussian_noise',
            'gaussian_blur',
            'jpeg_compression',
            'rotate',
            'affine_transformation'
        ]

        # ��֤���������Ƿ���Ч
        if apply_attack and attack_method is not None and attack_method not in self.valid_attack_methods:
            raise ValueError(f"Invalid: {attack_method}. valid: {self.valid_attack_methods}")

    def collect_image(self, root):
        image_path_list = []
        for split in os.listdir(root):
            split_root = os.path.join(root, split)
            if not os.path.isdir(split_root):
                continue
            img_list = os.listdir(split_root)
            # frame_num
            img_list = img_list if len(img_list) < self.frame_num else img_list[:self.frame_num]
            for img in img_list:
                img_path = os.path.join(split_root, img)
                image_path_list.append(img_path)
        return image_path_list

    def read_image(self, path):

        img = Image.open(path).convert('RGB')  # ȷ��ͼ��ΪRGB��ʽ
        return img

    def resize_image(self, image, size):

        img = image.resize((size, size))
        return img

    def apply_image_attack(self, img, attack_method):

        # PIL to numpy
        img_np = np.array(img)
        # ֱImageAttacks
        attacker = ImageAttacks(img=img_np)

        attack_func = getattr(attacker, attack_method)
        attacked_img_np = attack_func(self.attack_severity)

        # numpy to PIL
        attacked_img = Image.fromarray(attacked_img_np)
        return attacked_img

    def __getitem__(self, index):

        image_path = self.train_list[index]
        img = self.read_image(image_path)
        img = self.resize_image(img, size=self.size)

        if self.apply_attack and self.attack_method is not None:
            img = self.apply_image_attack(img, self.attack_method)

        img = self.transform(img)

        return img

    def __len__(self):

        return len(self.train_list)
        
a = 5



def get_dataset(name='train', size=224, root='/home/admin1/DFMamba/dataset/FF++c23/', frame_num=32, augment=True, apply_attack=False, attack_severity=a, attack_method='affine_transformation'):
    root = os.path.join(root, name)
    fake_root = os.path.join(root, 'fake')
    # fake_list = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    # Celeb-synthesis
    # wilddeepfake
    # DFDC
    
    fake_list = ['NeuralTextures']

    total_len = len(fake_list)
    dset_lst = []
    for i in range(total_len):
        fake = os.path.join(fake_root, fake_list[i])
        dset = FFDataset(fake, frame_num, size, augment, apply_attack, attack_severity, attack_method)
        dset.size = size
        dset_lst.append(dset)

    return torch.utils.data.ConcatDataset(dset_lst), total_len
    
def evaluate(model, data_path, mode='test'):
    root = data_path
    origin_root = root
    root = os.path.join(data_path, mode)
    real_root = os.path.join(root, 'real')
    dataset_real = FFDataset(dataset_root=real_root, size=224, frame_num=32, augment=False, apply_attack=False, attack_severity=a, attack_method='affine_transformation')
    dataset_fake, _ = get_dataset(name=mode, root=origin_root, size=224, frame_num=32, augment=False)
    dataset_img = torch.utils.data.ConcatDataset([dataset_real, dataset_fake])

    bz = 32
    # torch.cache.empty_cache()
    labels = []
    with torch.no_grad(): 
        y_true, y_pred = [], []

        for i, d in enumerate(dataset_img.datasets):
            dataloader = torch.utils.data.DataLoader(
                dataset=d,
                batch_size=bz,
                shuffle=True,
                num_workers=0,
                drop_last=True
                
            )
            for img in dataloader:
                if i == 0:
                    label = torch.zeros(img.size(0))
                else:
                    label = torch.ones(img.size(0))
                    
                labels.append(label.cpu().numpy())
                
                img = img.detach().cuda()
                output = model.forward(img)
                y_pred.extend(output.sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    AUC = cal_auc(fpr, tpr)

    idx_real = np.where(y_true == 0)[0]
    idx_fake = np.where(y_true == 1)[0]

    r_acc = accuracy_score(y_true[idx_real], y_pred[idx_real] > 0.5)
    f_acc = accuracy_score(y_true[idx_fake], y_pred[idx_fake] > 0.5)
    t_acc = accuracy_score(y_true, y_pred > 0.5)
    
    #np.save("/home/admin1/DFMamba/dataset_tsne/DFM_nt_labels.npy", labels)
    #np.save('/home/admin1/DFMamba/temp/DFM_nt_score.npy', y_pred)

    return AUC, r_acc, f_acc, t_acc
    
    
#########################################################################################    
def get_dataset1(name='train', size=224, root='/home/admin1/DFMamba/dataset/FF++c23/', frame_num=32, augment=True):
    root = os.path.join(root, name)
    fake_root = os.path.join(root, 'fake')
    # fake_list = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    fake_list = ['Deepfakes']

    total_len = len(fake_list)
    dset_lst = []
    for i in range(total_len):
        fake = os.path.join(fake_root, fake_list[i])
        dset = FFDataset(fake, frame_num, size, augment)
        dset.size = size
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst), total_len
    
def evaluate1(model, data_path, mode='test'):
    root = data_path
    origin_root = root
    root = os.path.join(data_path, mode)
    real_root = os.path.join(root, 'real')
    dataset_real = FFDataset(dataset_root=real_root, size=224, frame_num=32, augment=False)
    dataset_fake, _ = get_dataset1(name=mode, root=origin_root, size=224, frame_num=32, augment=False)
    dataset_img = torch.utils.data.ConcatDataset([dataset_real, dataset_fake])

    bz = 32
    # torch.cache.empty_cache()
    with torch.no_grad():
        y_true, y_pred = [], []

        for i, d in enumerate(dataset_img.datasets):
            dataloader = torch.utils.data.DataLoader(
                dataset=d,
                batch_size=bz,
                shuffle=True,
                num_workers=0
            )
            for img in dataloader:
                if i == 0:
                    label = torch.zeros(img.size(0))
                else:
                    label = torch.ones(img.size(0))
                img = img.detach().cuda()
                output = model.forward(img)
                y_pred.extend(output.sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    AUC = cal_auc(fpr, tpr)

    idx_real = np.where(y_true == 0)[0]
    idx_fake = np.where(y_true == 1)[0]

    r_acc = accuracy_score(y_true[idx_real], y_pred[idx_real] > 0.5)
    f_acc = accuracy_score(y_true[idx_fake], y_pred[idx_fake] > 0.5)
    t_acc = accuracy_score(y_true, y_pred > 0.5)

    return AUC, r_acc, f_acc, t_acc
#########################################################################################  

def get_dataset2(name='train', size=224, root='/home/admin1/DFMamba/dataset/FF++c23/', frame_num=32, augment=True):
    root = os.path.join(root, name)
    fake_root = os.path.join(root, 'fake')
    # fake_list = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    fake_list = ['Face2Face']

    total_len = len(fake_list)
    dset_lst = []
    for i in range(total_len):
        fake = os.path.join(fake_root, fake_list[i])
        dset = FFDataset(fake, frame_num, size, augment)
        dset.size = size
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst), total_len
    
    
def evaluate2(model, data_path, mode='test'):
    root = data_path
    origin_root = root
    root = os.path.join(data_path, mode)
    real_root = os.path.join(root, 'real')
    dataset_real = FFDataset(dataset_root=real_root, size=224, frame_num=32, augment=False)
    dataset_fake, _ = get_dataset2(name=mode, root=origin_root, size=224, frame_num=32, augment=False)
    dataset_img = torch.utils.data.ConcatDataset([dataset_real, dataset_fake])

    bz = 32
    # torch.cache.empty_cache()
    with torch.no_grad():
        y_true, y_pred = [], []

        for i, d in enumerate(dataset_img.datasets):
            dataloader = torch.utils.data.DataLoader(
                dataset=d,
                batch_size=bz,
                shuffle=True,
                num_workers=0
            )
            for img in dataloader:
                if i == 0:
                    label = torch.zeros(img.size(0))
                else:
                    label = torch.ones(img.size(0))
                img = img.detach().cuda()
                output = model.forward(img)
                y_pred.extend(output.sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    AUC = cal_auc(fpr, tpr)

    idx_real = np.where(y_true == 0)[0]
    idx_fake = np.where(y_true == 1)[0]

    r_acc = accuracy_score(y_true[idx_real], y_pred[idx_real] > 0.5)
    f_acc = accuracy_score(y_true[idx_fake], y_pred[idx_fake] > 0.5)
    t_acc = accuracy_score(y_true, y_pred > 0.5)

    return AUC, r_acc, f_acc, t_acc
    
#########################################################################################  
    
def get_dataset3(name='train', size=224, root='/home/admin1/DFMamba/dataset/FF++c23/', frame_num=20, augment=True):
    root = os.path.join(root, name)
    fake_root = os.path.join(root, 'fake')
    # fake_list = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    fake_list = ['FaceSwap']

    total_len = len(fake_list)
    dset_lst = []
    for i in range(total_len):
        fake = os.path.join(fake_root, fake_list[i])
        dset = FFDataset(fake, frame_num, size, augment)
        dset.size = size
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst), total_len
    
def evaluate3(model, data_path, mode='test'):
    root = data_path
    origin_root = root
    root = os.path.join(data_path, mode)
    real_root = os.path.join(root, 'real')
    dataset_real = FFDataset(dataset_root=real_root, size=224, frame_num=32, augment=False)
    dataset_fake, _ = get_dataset3(name=mode, root=origin_root, size=224, frame_num=32, augment=False)
    dataset_img = torch.utils.data.ConcatDataset([dataset_real, dataset_fake])

    bz = 32
    # torch.cache.empty_cache()
    with torch.no_grad():
        y_true, y_pred = [], []

        for i, d in enumerate(dataset_img.datasets):
            dataloader = torch.utils.data.DataLoader(
                dataset=d,
                batch_size=bz,
                shuffle=True,
                num_workers=0
            )
            for img in dataloader:
                if i == 0:
                    label = torch.zeros(img.size(0))
                else:
                    label = torch.ones(img.size(0))
                img = img.detach().cuda()
                output = model.forward(img)
                y_pred.extend(output.sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    AUC = cal_auc(fpr, tpr)

    idx_real = np.where(y_true == 0)[0]
    idx_fake = np.where(y_true == 1)[0]

    r_acc = accuracy_score(y_true[idx_real], y_pred[idx_real] > 0.5)
    f_acc = accuracy_score(y_true[idx_fake], y_pred[idx_fake] > 0.5)
    t_acc = accuracy_score(y_true, y_pred > 0.5)

    return AUC, r_acc, f_acc, t_acc
#########################################################################################  
    
def get_dataset4(name='train', size=224, root="/home/admin1/DFMamba/dataset/FF++c40/", frame_num=20, augment=True):
    root = os.path.join(root, name)
    fake_root = os.path.join(root, 'fake')
    # fake_list = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    fake_list = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

    total_len = len(fake_list)
    dset_lst = []
    for i in range(total_len):
        fake = os.path.join(fake_root, fake_list[i])
        dset = FFDataset(fake, frame_num, size, augment)
        dset.size = size
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst), total_len


def evaluate4(model, data_path, mode='test'):
    root = data_path
    origin_root = root
    root = os.path.join(data_path, mode)
    real_root = os.path.join(root, 'real')
    dataset_real = FFDataset(dataset_root=real_root, size=224, frame_num=32, augment=False)
    dataset_fake, _ = get_dataset4(name=mode, root=origin_root, size=224, frame_num=32, augment=False)
    dataset_img = torch.utils.data.ConcatDataset([dataset_real, dataset_fake])

    bz = 32
    # torch.cache.empty_cache()
    with torch.no_grad():
        y_true, y_pred = [], []

        for i, d in enumerate(dataset_img.datasets):
            dataloader = torch.utils.data.DataLoader(
                dataset=d,
                batch_size=bz,
                shuffle=True,
                num_workers=0
            )
            for img in dataloader:
                if i == 0:
                    label = torch.zeros(img.size(0))
                else:
                    label = torch.ones(img.size(0))
                img = img.detach().cuda()
                output = model.forward(img)
                y_pred.extend(output.sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    AUC = cal_auc(fpr, tpr)

    idx_real = np.where(y_true == 0)[0]
    idx_fake = np.where(y_true == 1)[0]

    r_acc = accuracy_score(y_true[idx_real], y_pred[idx_real] > 0.5)
    f_acc = accuracy_score(y_true[idx_fake], y_pred[idx_fake] > 0.5)
    t_acc = accuracy_score(y_true, y_pred > 0.5)

    return AUC, r_acc, f_acc, t_acc
    
#########################################################################################  
    
def get_dataset5(name='train', size=224, root="/home/admin1/DFMamba/dataset/CelebDF-v2/", frame_num=20, augment=True):
    root = os.path.join(root, name)
    fake_root = os.path.join(root, 'fake')
    fake_list = ['Celeb-synthesis']

    total_len = len(fake_list)
    dset_lst = []
    for i in range(total_len):
        fake = os.path.join(fake_root, fake_list[i])
        dset = FFDataset(fake, frame_num, size, augment)
        dset.size = size
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst), total_len


def evaluate5(model, data_path, mode='test'):
    root = data_path
    origin_root = root
    root = os.path.join(data_path, mode)
    real_root = os.path.join(root, 'real')
    dataset_real = FFDataset(dataset_root=real_root, size=224, frame_num=32, augment=False)
    dataset_fake, _ = get_dataset5(name=mode, root=origin_root, size=224, frame_num=32, augment=False)
    dataset_img = torch.utils.data.ConcatDataset([dataset_real, dataset_fake])

    bz = 32
    # torch.cache.empty_cache()
    with torch.no_grad():
        y_true, y_pred = [], []

        for i, d in enumerate(dataset_img.datasets):
            dataloader = torch.utils.data.DataLoader(
                dataset=d,
                batch_size=bz,
                shuffle=True,
                num_workers=0
            )
            for img in dataloader:
                if i == 0:
                    label = torch.zeros(img.size(0))
                else:
                    label = torch.ones(img.size(0))
                img = img.detach().cuda()
                output = model.forward(img)
                y_pred.extend(output.sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    AUC = cal_auc(fpr, tpr)

    idx_real = np.where(y_true == 0)[0]
    idx_fake = np.where(y_true == 1)[0]

    r_acc = accuracy_score(y_true[idx_real], y_pred[idx_real] > 0.5)
    f_acc = accuracy_score(y_true[idx_fake], y_pred[idx_fake] > 0.5)
    t_acc = accuracy_score(y_true, y_pred > 0.5)

    return AUC, r_acc, f_acc, t_acc
    
#########################################################################################  
    
def get_dataset6(name='train', size=224, root="/home/admin1/DFMamba/dataset/WildDeepfake/", frame_num=20, augment=True):
    root = os.path.join(root, name)
    fake_root = os.path.join(root, 'fake')
    fake_list = ['wilddeepfake']

    total_len = len(fake_list)
    dset_lst = []
    for i in range(total_len):
        fake = os.path.join(fake_root, fake_list[i])
        dset = FFDataset(fake, frame_num, size, augment)
        dset.size = size
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst), total_len


def evaluate6(model, data_path, mode='test'):
    root = data_path
    origin_root = root
    root = os.path.join(data_path, mode)
    real_root = os.path.join(root, 'real')
    dataset_real = FFDataset(dataset_root=real_root, size=224, frame_num=32, augment=False)
    dataset_fake, _ = get_dataset6(name=mode, root=origin_root, size=224, frame_num=32, augment=False)
    dataset_img = torch.utils.data.ConcatDataset([dataset_real, dataset_fake])

    bz = 32
    # torch.cache.empty_cache()
    with torch.no_grad():
        y_true, y_pred = [], []

        for i, d in enumerate(dataset_img.datasets):
            dataloader = torch.utils.data.DataLoader(
                dataset=d,
                batch_size=bz,
                shuffle=True,
                num_workers=0
            )
            for img in dataloader:
                if i == 0:
                    label = torch.zeros(img.size(0))
                else:
                    label = torch.ones(img.size(0))
                img = img.detach().cuda()
                output = model.forward(img)
                y_pred.extend(output.sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    AUC = cal_auc(fpr, tpr)

    idx_real = np.where(y_true == 0)[0]
    idx_fake = np.where(y_true == 1)[0]

    r_acc = accuracy_score(y_true[idx_real], y_pred[idx_real] > 0.5)
    f_acc = accuracy_score(y_true[idx_fake], y_pred[idx_fake] > 0.5)
    t_acc = accuracy_score(y_true, y_pred > 0.5)

    return AUC, r_acc, f_acc, t_acc
      
#########################################################################################  
    
def get_dataset7(name='train', size=224, root="/home/admin1/DFMamba/dataset/DFDCp/", frame_num=32, augment=True):
    root = os.path.join(root, name)
    fake_root = os.path.join(root, 'fake')
    fake_list = ['DFDC']

    total_len = len(fake_list)
    dset_lst = []
    for i in range(total_len):
        fake = os.path.join(fake_root, fake_list[i])
        dset = FFDataset(fake, frame_num, size, augment)
        dset.size = size
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst), total_len


def evaluate7(model, data_path, mode='test'):
    root = data_path
    origin_root = root
    root = os.path.join(data_path, mode)
    real_root = os.path.join(root, 'real')
    dataset_real = FFDataset(dataset_root=real_root, size=224, frame_num=32, augment=False)
    dataset_fake, _ = get_dataset7(name=mode, root=origin_root, size=224, frame_num=32, augment=False)
    dataset_img = torch.utils.data.ConcatDataset([dataset_real, dataset_fake])

    bz = 32
    # torch.cache.empty_cache()
    with torch.no_grad():
        y_true, y_pred = [], []

        for i, d in enumerate(dataset_img.datasets):
            dataloader = torch.utils.data.DataLoader(
                dataset=d,
                batch_size=bz,
                shuffle=True,
                num_workers=0
            )
            for img in dataloader:
                if i == 0:
                    label = torch.zeros(img.size(0))
                else:
                    label = torch.ones(img.size(0))
                img = img.detach().cuda()
                output = model.forward(img)
                y_pred.extend(output.sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    AUC = cal_auc(fpr, tpr)

    idx_real = np.where(y_true == 0)[0]
    idx_fake = np.where(y_true == 1)[0]

    r_acc = accuracy_score(y_true[idx_real], y_pred[idx_real] > 0.5)
    f_acc = accuracy_score(y_true[idx_fake], y_pred[idx_fake] > 0.5)
    t_acc = accuracy_score(y_true, y_pred > 0.5)

    return AUC, r_acc, f_acc, t_acc
    

    


# python 3.7
"""Utility functions for logging."""

__all__ = ['setup_logger']

DEFAULT_WORK_DIR = 'results'


def setup_logger(work_dir=None, logfile_name='log.txt', logger_name='logger'):
    """Sets up logger from target work directory.

    The function will sets up a logger with `DEBUG` log level. Two handlers will
    be added to the logger automatically. One is the `sys.stdout` stream, with
    `INFO` log level, which will print improtant messages on the screen. The other
    is used to save all messages to file `$WORK_DIR/$LOGFILE_NAME`. Messages will
    be added time stamp and log level before logged.

    NOTE: If `logfile_name` is empty, the file stream will be skipped. Also,
    `DEFAULT_WORK_DIR` will be used as default work directory.

    Args:
    work_dir: The work directory. All intermediate files will be saved here.
        (default: None)
    logfile_name: Name of the file to save log message. (default: `log.txt`)
    logger_name: Unique name for the logger. (default: `logger`)

    Returns:
    A `logging.Logger` object.

    Raises:
    SystemExit: If the work directory has already existed, of the logger with
        specified name `logger_name` has already existed.
    """

    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():  # Already existed
        raise SystemExit(f'Logger name `{logger_name}` has already been set up!\n'
                         f'Please use another name, or otherwise the messages '
                         f'may be mixed between these two loggers.')

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    # Print log message with `INFO` level or above onto the screen.
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if not logfile_name:
        return logger

    work_dir = work_dir or DEFAULT_WORK_DIR
    logfile_name = os.path.join(work_dir, logfile_name)
    # if os.path.isfile(logfile_name):
    #   print(f'Log file `{logfile_name}` has already existed!')
    #   while True:
    #     decision = input(f'Would you like to overwrite it (Y/N): ')
    #     decision = decision.strip().lower()
    #     if decision == 'n':
    #       raise SystemExit(f'Please specify another one.')
    #     if decision == 'y':
    #       logger.warning(f'Overwriting log file `{logfile_name}`!')
    #       break

    os.makedirs(work_dir, exist_ok=True)

    # Save log message with all levels in log file.
    fh = logging.FileHandler(logfile_name)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
