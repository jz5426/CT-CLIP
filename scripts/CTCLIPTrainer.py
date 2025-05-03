from pathlib import Path
from datetime import timedelta

from transformer_maskgit.optimizer import get_optimizer
from transformers import BertTokenizer

from eval import evaluate_internal
from sklearn.metrics import f1_score, accuracy_score


import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from data import CustomCTReportDataset

import numpy as np
import pandas as pd

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

import math
import torch.optim.lr_scheduler as lr_scheduler
from ct_clip import CTCLIP
import os
import random


# helpers
def apply_softmax(array):
    """
    Applies softmax function to a torch array.

    Args:
        array (torch.Tensor): Input tensor array.

    Returns:
        torch.Tensor: Tensor array after applying softmax.
    """
    softmax = torch.nn.Softmax(dim=0)
    softmax_array = softmax(array)
    return softmax_array

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_warmup=10000, gamma=1.0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.T_warmup = T_warmup
        self.gamma = gamma
        self.T_cur = 0
        self.lr_min = 0
        self.iteration = 0

        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.iteration < self.T_warmup:
            lr = self.eta_max * self.iteration / self.T_warmup
        else:
            self.T_cur = self.iteration - self.T_warmup
            T_i = self.T_0
            while self.T_cur >= T_i:
                self.T_cur -= T_i
                T_i *= self.T_mult
                self.lr_min = self.eta_max * (self.gamma ** self.T_cur)
            lr = self.lr_min + 0.5 * (self.eta_max - self.lr_min) * \
                 (1 + math.cos(math.pi * self.T_cur / T_i))

        self.iteration += 1
        return [lr for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self._update_lr()
        self._update_T()

    def _update_lr(self):
        self.optimizer.param_groups[0]['lr'] = self.get_lr()[0]

    def _update_T(self):
        if self.T_cur == self.T_0:
            self.T_cur = 0
            self.lr_min = 0
            self.iteration = 0
            self.T_0 *= self.T_mult
            self.eta_max *= self.gamma

class UniqueLevelSampler(Sampler):
    def __init__(self, key_ids, batch_size):
        """
        Args:
            patient_ids (list): List of patient IDs.
            batch_size (int): Number of unique patients per batch.
        """
        self.key_ids = key_ids
        self.batch_size = batch_size
    
    def __iter__(self):
        shuffled_ids = random.sample(self.key_ids, len(self.key_ids))
        for i in range(0, len(shuffled_ids), self.batch_size):
            yield shuffled_ids[i:i + self.batch_size]
    
    def __len__(self):
        return len(self.key_ids) // self.batch_size


class CTClipTrainer(nn.Module):
    def __init__(
        self,
        CTClip: CTCLIP,
        *,
        min_epochs,
        batch_size,
        meta_data='',
        batch_style='patient',
        data_train = "train",
        data_valid = "valid",
        reports_file_train = "data_reports.xslx",
        reports_file_valid = "data_reports.xslx",
        labels = "labels.csv",
        tokenizer = None,
        lr = 5e-5, # TODO: double check the original CTCLIPTrainer parameters
        wd = 1e-4, # TODO: double check the original CTCLIPTrainer parameters
        max_grad_norm = 0.5,
        iteration_evaluate_frequency = 2,
        epoch_based_patience = 10,
        save_results_every = 1000,
        save_model_every = 1000 ,
        results_folder = '',
        num_workers = 8,
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, kwargs], **accelerate_kwargs)
        self.CTClip = CTClip
        self.min_epochs = min_epochs

        self.max_grad_norm = max_grad_norm
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
        self.register_buffer('steps', torch.Tensor([0]))

        self.batch_size = batch_size
        all_parameters = set(CTClip.parameters())

        self.optim = get_optimizer(all_parameters, lr=lr, wd=wd, group_wd_params=False)
        self.lr=lr
        
        self.dl = None
        self.valid_dl = None

        # the following is for our preprocessed CT data.
        #TODO: potentially need to implement patient based or experiment based contrastive learning like x2ct-clip
        self.train_ds = CustomCTReportDataset(
            data_folder=data_train, 
            csv_file=reports_file_train,
            meta_data=pd.read_csv(meta_data),
            split='train'
        )
        self.valid_ds = CustomCTReportDataset(
            data_folder=data_valid, 
            csv_file=reports_file_valid,
            label_file=labels,
            meta_data=pd.read_csv(meta_data),
            split='val'
        )

        self.dl = DataLoader(
            self.train_ds,
            num_workers=num_workers,
            batch_size=self.batch_size,
            shuffle = True
        )

        self.valid_dl = DataLoader(
            self.valid_ds,
            num_workers=num_workers,
            batch_size=self.batch_size,
            shuffle = False
        )

        # prepare with accelerator
        self.dl_iter, self.valid_dl_iter = None, None
        if self.dl:
            self.dl_iter=cycle(self.dl)
        if self.valid_dl:
            self.valid_dl_iter=cycle(self.valid_dl)
        self.device = self.accelerator.device
        self.CTClip.to(self.device)
        
        (
 			self.dl_iter,
            self.valid_dl_iter,
            self.CTClip,
            self.optim,
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.valid_dl_iter,
            self.CTClip,
            self.optim,
        )

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.iteration_evaluate_frequency = iteration_evaluate_frequency
        self.epoch_based_patience = epoch_based_patience
        self.early_stop_counter = 0
        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.best_flat_val_acc = 0
        self.best_f1_val_acc = 0
        self.best_iter_based_val_cl_loss = float('inf')
        self.best_epoch_based_val_cl_loss = float('inf')

        # base file name for the checkpoints
        self.base_file_name = f'modeltype_ctclip__batchstyle_{batch_style}__bs_{batch_size}__lr_{lr}__wd_{wd}'
        print('base file name: ', self.base_file_name)

    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model=self.accelerator.get_state_dict(self.CTClip),
            optim=self.optim.state_dict(),
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)

        CTClip = self.accelerator.unwrap_model(self.CTClip)
        CTClip.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        self.accelerator.print(msg)


    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def train_by_epoch(self, epochs):
        print('Epoch Training Starts\n')
        device = self.device

        # in unit of batch size
        train_size = len(self.dl)
        val_size = len(self.valid_dl) if self.valid_dl else 0

        for epoch in range(epochs):
            self.CTClip.train()
            running_loss = 0.0
            for batch_idx in range(train_size):
                self.optim.zero_grad()

                data = next(self.dl_iter)
                video, text = data['ct'], data['report']
                video=video.to(device)

                with self.accelerator.autocast(): # forward pass of triplet ct_clip model.
                    text = list(text)
                    text_tokens=self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device) # automatically prepend the [CLS] token with id 2, 511 actual content maximum.
                    loss = self.CTClip(text_tokens, video, return_loss=True, device=device)

                self.accelerator.backward(loss)
                if exists(self.max_grad_norm): # NOTE: should i keep the gradient clip during training.
                    self.accelerator.clip_grad_norm_(self.CTClip.parameters(), self.max_grad_norm)
                self.optim.step()

                # evaluate model based on iteration instead of epochs
                if self.is_main and not (batch_idx % self.iteration_evaluate_frequency):
                    print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{train_size}] in training split, Training Loss: {loss.item():.4f}")

                # Accumulate loss
                running_loss += loss.item()

            # run per-epoch validation and automatically save the model
            if val_size > 0:
                print(f'Validation after epoch {epoch}')
                exit_training = self.eval_on_validation_split(epoch, val_size)

            # Print average loss for the epoch
            epoch_loss = running_loss / train_size
            print(f"Epoch [{epoch+1}/{epochs}] completed with average training loss: {epoch_loss:.4f}")

            if exit_training:
                print('Training by epochs complete\n')
                return

        print('Training by epochs complete\n')

    def eval_on_validation_split(self, epoch, val_size, iteration=-1):
        """
        return: boolean -> whether should stop training or nort.
        """
        device = self.device
        # after training each epoch, test the model in the validation split
        if self.is_main:
            with torch.no_grad():
                self.CTClip.eval()
                predictedall=[]
                realall=[]
                running_val_loss = 0
                for i in range(val_size): #NOTE: might need to change this to evaluate on the whole validation set.
                    val_data = next(self.valid_dl_iter)
                    valid_data, text, onehotlabels = val_data['ct'], val_data['report'], val_data['label']
                    valid_data = valid_data.to(device)

                    # mainly for the validation contrastive loss
                    report_tokens=self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
                    val_cl_loss = self.CTClip(report_tokens, valid_data, return_loss=True, device=device)

                    # Accumulate validation contrastive loss for this epochs
                    running_val_loss += val_cl_loss.item()

                    print(f"    Evaluating Batch {i}/{val_size} in validation split")

                    if "module" in self.CTClip.__dict__:
                        self.CTClip = self.CTClip.module

                    pathologies = ['Medical material',
                                    'Arterial wall calcification', 
                                    'Cardiomegaly', 
                                    'Pericardial effusion',
                                    'Coronary artery wall calcification', 
                                    'Hiatal hernia',
                                    'Lymphadenopathy', 
                                    'Emphysema', 
                                    'Atelectasis', 
                                    'Lung nodule',
                                    'Lung opacity', 
                                    'Pulmonary fibrotic sequela', 
                                    'Pleural effusion', 
                                    'Mosaic attenuation pattern',
                                    'Peribronchial thickening', 
                                    'Consolidation', 
                                    'Bronchiectasis',
                                    'Interlobular septal thickening']
                    plotdir = str(self.results_folder / f'CTClip_{epoch}' )
                    plotdir = plotdir + os.sep

                    Path(plotdir).mkdir(parents=True, exist_ok=True)

                    predictedlabels = [[] for _ in range(onehotlabels.shape[0])] # hold the predicted multi-label vector for each sample in the batch
                    for pathology in pathologies:
                        text = [f"There is {pathology}.", f"There is no {pathology}."] #NOTE: binary classification for each pathology.
                        text_tokens=self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

                        # this should be the logit score between the text and xray
                        logits = self.CTClip(text_tokens, valid_data, device=device)

                        outputs = apply_softmax(logits)

                        for idx in range(outputs.shape[-1]): # batch size
                            output = outputs[:,idx]
                            if output[0]>output[1]:
                                predictedlabels[idx].append(1) # 1 indicates has pathology in the one-hot label
                            else:
                                predictedlabels[idx].append(0) # 0 indicates no pathnology in the one-hot label
                    
                    # append the pathology classifications for one validation image
                    predictedall.extend(predictedlabels)
                    realall.extend(onehotlabels.detach().cpu().tolist())

                # Print and save classification report
                realall=np.array(realall)
                predictedall=np.array(predictedall)

                dfs=evaluate_internal(predictedall, realall, pathologies, plotdir)
                realall = np.rint(realall).astype(int)
                predictedall = np.rint(predictedall).astype(int)
                
                f1 = f1_score(realall, predictedall,average='micro')
                flat_acc = accuracy_score(realall.flatten(), predictedall.flatten())
                print('    Validation F1 Accuracy: {}; Validation Flat Accuracy: {}\n'.format(f1, flat_acc))
                # NOTE: high flat accuracy but low f1 accuracy indicates poor minority class performance
                writer = pd.ExcelWriter(f'{plotdir}aurocs.xlsx', engine='xlsxwriter')

                dfs.to_excel(writer, sheet_name='Sheet1', index=False)
                writer.close()
                del output

                #NOTE save model based on predefined epoch and always saving the last epoch
                # self._save_ckpt(epoch, 'last_epoch.pt', 'saving the last epoch checkpoint', iteration)
                # if epoch % self.min_epochs == 0:
                #     self._save_ckpt(epoch, f'{epoch}_epoch.pt', f'saving the {epoch}th epoch checkpoint', iteration)

                # save checkpoint for every epoch
                self._save_ckpt(epoch, 'checkpoint_{}.pt'.format(epoch), 'saving the epoch checkpoint', iteration)

        return False

    def _save_ckpt(self, epoch, file_name, print_annotation, iteration=-1):
        """
        iteration = -1 indicates this is epoch based training/evaluation
        iteration != -1 indicates this is interation based training/evaluation 
        """
        model_path = str(self.results_folder / f'{self.base_file_name}_{file_name}')
        state_dict=self.accelerator.get_state_dict(self.CTClip, unwrap=False)
        self.accelerator.save(state_dict, model_path)
        if iteration == -1:
            print(f'    Epoch:{epoch}: saving model to {str(self.results_folder)} -- {print_annotation}\n')
            return
        print(f'    Epoch:{epoch} - iteration:{iteration}: saving model to {str(self.results_folder)} -- {print_annotation}\n')