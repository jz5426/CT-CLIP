from pathlib import Path
from shutil import rmtree
from datetime import timedelta

from transformer_maskgit.optimizer import get_optimizer
from transformers import BertTokenizer, BertModel

from eval import evaluate_internal, plot_roc, accuracy, sigmoid, bootstrap, compute_cis
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split, Sampler
from torch.utils.data.distributed import DistributedSampler

from data import CTReportDataset, CTReportXRayDataset
from data_inference import CTReportDatasetinfer, CTReportXRayDatasetinfer

import numpy as np
import pandas as pd

from einops import rearrange
import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

import math
import torch.optim.lr_scheduler as lr_scheduler
from ct_clip import CTCLIP
import os
import random
import tqdm


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



def tensor_to_nifti(tensor, path, affine=np.eye(4)):
    """
    Save tensor as a NIfTI file.

    Args:
        tensor (torch.Tensor): The input tensor with shape (D, H, W) or (C, D, H, W).
        path (str): The path to save the NIfTI file.
        affine (np.ndarray, optional): The affine matrix for the NIfTI file. Defaults to np.eye(4).
    """

    tensor = tensor.cpu()

    if tensor.dim() == 4:
        # Assume single channel data if there are multiple channels
        if tensor.size(0) != 1:
            print("Warning: Saving only the first channel of the input tensor")
        tensor = tensor.squeeze(0)
    tensor=tensor.swapaxes(0,2)
    numpy_data = tensor.detach().numpy().astype(np.float32)
    nifti_img = nib.Nifti1Image(numpy_data, affine)
    nib.save(nifti_img, path)

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

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
        num_train_steps,
        batch_size,
        text_cl_weight = 1.,
        ct_cl_weight = 1.,
        batch_style='patient',
        data_train = "train",
        data_valid = "valid",
        cfg=None,
        img_embedding_paths = {}, # contain both train and validation
        text_embedding_paths = {}, # contian both train and validation
        reports_file_train = "data_reports.xslx",
        reports_file_valid = "data_reports.xslx",
        labels = "labels.csv",
        tokenizer = None,
        lr = 5e-5, # 1.25e-6, suggested by ULIP, 5e-5 from cxr-clip
        wd = 1e-4, # NOTE: from cxr-clip
        max_grad_norm = 0.5,
        iteration_evaluate_frequency = 2,
        epoch_based_patience = 10,
        save_results_every = 1000,
        save_model_every = 1000 ,
        results_folder = '',
        num_workers = 8,
        train_from_scratch = True, # TODO: double check this!
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, kwargs], **accelerate_kwargs)
        self.CTClip = CTClip

        # NOTE: automatic toggle: alter to ULIP-style mode if the xray encoder exists in the CTCLIP
        self.triplet = False
        if hasattr(self.CTClip, 'xray_encoder'):
            self.triplet = True
            # max_grad_norm = None # TODO: might need to experiment if need this.

        self.max_grad_norm = max_grad_norm
        
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        all_parameters = set(CTClip.parameters())

        #TODO: might need to check if group_wd_params is needed when train the whole triplet instead of the xray encoder.
        if cfg and cfg['optimizer']['name'] == 'adamw':
            wd = cfg['optimizer']['config']['weight_decay']
            lr = cfg['optimizer']['config']['lr']
        else:
            # default parameters in original CTCLIPTrainer
            wd = 0
            lr = 1.25e-6

        self.optim = get_optimizer(all_parameters, lr=lr, wd=wd, group_wd_params=False)
        self.lr=lr
        
        # Load the pre-trained weights
        self.img_embedding_paths = img_embedding_paths
        self.text_embedding_paths = text_embedding_paths
        if self.triplet:

            # train does not need csv file as it does not requires report
            self.train_ds = CTReportXRayDataset(
                data_folder=data_train, 
                cfg=cfg, 
                csv_file=reports_file_train,
                img_embedding_path=img_embedding_paths['train'], 
                text_embedding_path=text_embedding_paths['train'],
                batch_style=batch_style
            )
            self.valid_ds = CTReportXRayDatasetinfer(
                data_folder=data_valid, 
                cfg=cfg, 
                csv_file=reports_file_valid,
                img_embedding_path=img_embedding_paths['valid'],
                text_embedding_path=text_embedding_paths['valid'],
                batch_style=batch_style,
                labels=labels
            )

            # custom sampler
            custom_train_sampler = UniqueLevelSampler(self.train_ds.key_ids, self.batch_size)
            custom_val_sampler = UniqueLevelSampler(self.valid_ds.key_ids, self.batch_size)

            self.dl = DataLoader(
                self.train_ds,
                num_workers=num_workers,
                # shuffle = True,
                batch_sampler=custom_train_sampler
            )

            self.valid_dl = DataLoader(
                self.valid_ds,
                num_workers=num_workers,
                # shuffle = False,
                batch_sampler=custom_val_sampler
            )

        else:
            self.train_ds = CTReportDataset(data_folder=data_train, csv_file=reports_file_train)
            self.valid_ds = CTReportDatasetinfer(data_folder=data_valid, csv_file=reports_file_valid, labels=labels)

            self.dl = DataLoader(
                self.train_ds,
                num_workers=num_workers,
                batch_size=self.batch_size,
                shuffle = True,
            )

            self.valid_dl = DataLoader(
                self.valid_ds,
                num_workers=num_workers,
                batch_size=1,
                shuffle = False,
            )

        # prepare with accelerator
        self.dl_iter=cycle(self.dl)
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

        if len([*self.results_folder.glob('**/*')]) > 0 and train_from_scratch:
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.best_flat_val_acc = 0
        self.best_f1_val_acc = 0
        self.best_iter_based_val_cl_loss = float('inf')
        self.best_epoch_based_val_cl_loss = float('inf')
        self.text_cl_weight = text_cl_weight
        self.ct_cl_weight = ct_cl_weight

        # base file name for the checkpoints
        model_type = 'Swin' if cfg['model']['image_encoder']['model_type'] == 'swin' else 'Resnet'
        self.base_file_name = f'modeltype_{model_type}__batchstyle_{batch_style}__bs_{batch_size}__lr_{lr}__wd_{wd}__textcl_{self.text_cl_weight}__ctcl_{self.ct_cl_weight}'

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

    # def train_step(self):
    #     device = self.device
    #     steps = int(self.steps.item())

    #     self.CTClip.train() # set the models in train mode for gradient descent

    #     # logs
    #     logs = {}

    #     # update CTClip model
    #     data = next(self.dl_iter)
    #     if self.triplet_training:
    #         video, text, xray = data
    #         xray=xray.to(device)
    #     else:
    #         video, text = data

    #     print(video.shape)
        
    #     video=video.to(device)
    #     mask = torch.ones((video.shape[0], video.shape[2])).bool().to(device)
    #     text = list(text)
    #     text_tokens=self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device) # automatically prepend the [CLS] token with id 2, 511 actual content maximum.
    #     #NOTE: this is the actual forward pass of the CTCLIP
    #     with self.accelerator.autocast():
    #         if self.triplet_training:
    #             loss = self.CTClip(text_tokens, video, xray, return_loss=True, device=device)
    #         else:
    #             loss = self.CTClip(text_tokens, video, return_loss=True, device=device)

    #     self.accelerator.backward(loss)
    #     accum_log(logs, {'loss': loss.item()})
    #     if exists(self.max_grad_norm):
    #         self.accelerator.clip_grad_norm_(self.CTClip.parameters(), self.max_grad_norm)

    #     self.optim.step()
    #     self.optim.zero_grad()
    #     self.print(f"{steps}: loss: {logs['loss']}")

    #     if self.is_main and not (steps % self.save_results_every): # execute for every self.save_results_every
    #         with torch.no_grad():

    #             model = self.CTClip
    #             model.eval()
    #             predictedall=[]
    #             realall=[]

    #             #Fast inference on 100 images
    #             for i in range(10): #NOTE: might need to change this to evaluate on the whole validation set.
    #                 val_data = next(self.valid_dl_iter)

    #                 if self.triplet_training:
    #                     valid_data, text, onehotlabels, xray_image, _, _ = val_data
    #                     xray_image = xray_image.to(device)
    #                 else:
    #                     valid_data, text, onehotlabels, _, _ = val_data

    #                 valid_data = valid_data.to(device)

    #                 if "module" in model.__dict__:
    #                     model = model.module

    #                 pathologies = ['Medical material',
    #                                 'Arterial wall calcification', 
    #                                 'Cardiomegaly', 
    #                                 'Pericardial effusion',
    #                                 'Coronary artery wall calcification', 
    #                                 'Hiatal hernia',
    #                                 'Lymphadenopathy', 
    #                                 'Emphysema', 
    #                                 'Atelectasis', 
    #                                 'Lung nodule',
    #                                 'Lung opacity', 
    #                                 'Pulmonary fibrotic sequela', 
    #                                 'Pleural effusion', 
    #                                 'Mosaic attenuation pattern',
    #                                 'Peribronchial thickening', 
    #                                 'Consolidation', 
    #                                 'Bronchiectasis',
    #                                 'Interlobular septal thickening']
    #                 plotdir = str(self.results_folder / f'CTClip_{steps}' )
    #                 plotdir = plotdir + os.sep

    #                 Path(plotdir).mkdir(parents=True, exist_ok=True)

    #                 predictedlabels=[]
    #                 for pathology in pathologies:
    #                     text = [f"There is {pathology}.", f"There is no {pathology}."] #NOTE: binary classification for each pathology.
    #                     text_tokens=self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
                        
    #                     # this should be the logit score between the text and xray
    #                     logits = model(text_tokens, valid_data, xray_image, device=device, return_logits_only=True)
    #                     output = apply_softmax(logits)

    #                     if output[0]>output[1]:
    #                         predictedlabels.append(1) # 1 indicates has pathology in the one-hot label
    #                     else:
    #                         predictedlabels.append(0) # 0 indicates no pathnology in the one-hot label
                    
    #                 # append the pathology classifications for one validation image
    #                 predictedall.append(predictedlabels)
    #                 realall.append(onehotlabels.detach().cpu().numpy()[0])

    #             # Print and save classification report
    #             realall=np.array(realall)
    #             predictedall=np.array(predictedall)

    #             dfs=evaluate_internal(predictedall,realall,pathologies, plotdir)
    #             realall = np.rint(realall).astype(int)
    #             predictedall = np.rint(predictedall).astype(int)
                
    #             f1 = f1_score(realall, predictedall,average='micro')
    #             flat_acc = accuracy_score(realall.flatten(), predictedall.flatten())
    #             print('Test F1 Accuracy: ', f1)
    #             print('Test Flat Accuracy: ', flat_acc,'\n')
    #             # NOTE: high flat accuracy but low f1 accuracy indicates poor minority class performance
    #             writer = pd.ExcelWriter(f'{plotdir}aurocs.xlsx', engine='xlsxwriter')

    #             dfs.to_excel(writer, sheet_name='Sheet1', index=False)
    #             writer.close()
    #             del output

    #             # save model every so often (NOTE: should be based on epochs or iterations)
    #             model_path = str(self.results_folder / f'CTClip.{steps}.pt')
    #             state_dict=self.accelerator.get_state_dict(self.CTClip, unwrap=False)
    #             self.accelerator.save(state_dict, model_path)
    #             self.print(f'{steps}: saving model to {str(self.results_folder)}')
                
    #             if self.best_f1_val_acc < f1:
    #                 self.best_f1_val_acc = f1
    #                 model_path = str(self.results_folder / 'CTClip_best_f1_val.pt')
    #                 state_dict=self.accelerator.get_state_dict(self.CTClip, unwrap=False)
    #                 self.accelerator.save(state_dict, model_path)
    #                 self.print(f'{steps}: saving model to {str(self.results_folder)}')
                
    #             if self.best_flat_val_acc < flat_acc:
    #                 self.best_flat_val_acc = flat_acc
    #                 model_path = str(self.results_folder / 'CTClip_best_flat_acc_val.pt')
    #                 state_dict=self.accelerator.get_state_dict(self.CTClip, unwrap=False)
    #                 self.accelerator.save(state_dict, model_path)
    #                 self.print(f'{steps}: saving model to {str(self.results_folder)}')
                
    #     self.steps += 1
    #     return logs

    # def train(self, log_fn=noop):
    #     while self.steps < self.num_train_steps:
    #         logs = self.train_step()
    #         log_fn(logs)

    #     self.print('training complete')


    # def retrieval_evaluation(self, latent_type='ct', split='valid', topk=[1, 5, 10, 50]):

    #     # sanity check
    #     assert(split in ['valid', 'train'])
    #     assert(latent_type in ['ct', 'report'])
    #     assert(self.triplet == True)

    #     print('Retrieval Evaluation Starts\n')
    #     device = self.device
    #     data_size = len(self.valid_dl) if split == 'valid' else len(self.dl)
    #     data_iterator = self.valid_dl_iter if split == 'valid' else self.dl_iter

    #     # load the target embeddings for retrival.
    #     target_embedding_path = self.img_embedding_paths if latent_type == 'ct' else self.text_embedding_paths
    #     target_embedding_split_path = target_embedding_path['valid'] if split == 'valid' else target_embedding_path['train']
    #     target_embedding_dict = torch.load(target_embedding_split_path) # loaded from '.pth' file.

    #     # evaluation  mode
    #     with torch.no_grad():
    #         self.CTClip.eval()

    #         for batch_idx in range(data_size):
    #             data = next(data_iterator)
    #             video, text, xray = data
    #             xray=xray.to(device)
    #             text=text.to(device)
    #             video=video.to(device)

    #             # only get the xray latents (in the size of batch), in shape (batch, latent size)
    #             batch_xray_latents = self.CTClip.get_xray_latents(xray)

    #             # TODO: perform retrieval evaluation based on target_embedding_dict (agnostic to report or ct latents)


    #     return


    def train_by_epoch(self, epochs):
        print('Epoch Training Starts\n')
        device = self.device

        # in unit of batch size
        train_size = len(self.dl)
        val_size = len(self.valid_dl)

        for epoch in range(epochs):
            self.CTClip.train()
            running_loss = 0.0
            for batch_idx in range(train_size):
                self.optim.zero_grad()

                data = next(self.dl_iter)
                if self.triplet:
                    video, text, xray = data
                    xray=xray.to(device)
                    text=text.to(device)
                else:
                    video, text = data
                video=video.to(device)

                with self.accelerator.autocast(): # forward pass of triplet ct_clip model.
                    if self.triplet:
                        loss = self.CTClip(text,
                                           video, 
                                           xray, 
                                           device=device,
                                           text_cl_weight = self.text_cl_weight,
                                           ct_cl_weight = self.ct_cl_weight,
                                           is_text_latent_input=True, 
                                           is_image_latent_input=True)
                    else:
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
                    print('    Evaluate based on iterations')
                    self.eval_on_validation_split(epoch, val_size, iteration=batch_idx, is_epoch_evaluation=False)

                # Accumulate loss
                running_loss += loss.item()

            # run per-epoch validation and automatically save the model
            print(f'Validation after epoch {epoch}')
            exit_training = self.eval_on_validation_split(epoch, val_size, is_epoch_evaluation=True)

            # Print average loss for the epoch
            epoch_loss = running_loss / train_size
            print(f"Epoch [{epoch+1}/{epochs}] completed with average training loss: {epoch_loss:.4f}")

            if exit_training:
                print('Training by epochs complete\n')
                return

        print('Training by epochs complete\n')

    def eval_on_validation_split(self, epoch, val_size, iteration=-1, is_epoch_evaluation=False):
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

                    if self.triplet:
                        valid_data, text, onehotlabels, xray_image, _, _ = val_data
                        xray_image = xray_image.to(device)
                        text=text.to(device)
                    else:
                        valid_data, text, onehotlabels, _, _ = val_data

                    valid_data = valid_data.to(device)

                    # mainly for the validation contrastive loss
                    if self.triplet:
                        val_cl_loss = self.CTClip(text, 
                                                valid_data, 
                                                xray_image, 
                                                device=device, 
                                                text_cl_weight = self.text_cl_weight,
                                                ct_cl_weight = self.ct_cl_weight,
                                                is_text_latent_input=True, 
                                                is_image_latent_input=True)
                    else:
                        report_tokens=self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
                        val_cl_loss = self.CTClip(report_tokens, valid_data, xray_image, device=device)

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
                        if self.triplet:
                            logits = self.CTClip(text_tokens, 
                                                valid_data, 
                                                xray_image, 
                                                device=device, 
                                                is_text_latent_input=False, 
                                                is_image_latent_input=True,
                                                return_logits_only=True) # need this to return logits
                        else:
                            logits = self.CTClip(text_tokens, valid_data, xray_image, device=device, return_logits_only=True)

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

                # save model based on f1 metric
                if self.best_f1_val_acc < f1:
                    print(f'    Previous f1 {self.best_f1_val_acc} --> New best f1 {f1}')
                    self.best_f1_val_acc = f1
                    self._save_ckpt(epoch, 
                                    'CTClip_best_f1_val.pt', 
                                    'best f1 accuracy achieved!!', 
                                    iteration)
                
                # save model based on flat acc
                if self.best_flat_val_acc < flat_acc:
                    print(f'    Previous flat accuracy {self.best_flat_val_acc} --> New best accuracy {flat_acc}')
                    self.best_flat_val_acc = flat_acc
                    self._save_ckpt(epoch, 
                                    'CTClip_best_flat_acc_val.pt', 
                                    'best flat accuracy achieved!!', 
                                    iteration)

                # save model based on contrastive loss on validation split DURING ITERATION EVALUATION
                epoch_val_cl_loss = running_val_loss / val_size

                if not is_epoch_evaluation and self.best_iter_based_val_cl_loss > epoch_val_cl_loss:
                    print(f'    Iteration evaluation: Previous validation contrastive loss {self.best_iter_based_val_cl_loss} --> New validation contrastive loss {epoch_val_cl_loss}')
                    self.best_iter_based_val_cl_loss = epoch_val_cl_loss
                    self._save_ckpt(epoch, 
                                    'CTClip_lowest_val_cl_loss_during_iterations.pt', 
                                    'best contrastive loss on validation split!!', 
                                    iteration)

                # save model based on contrastive loss on validation split DURING EPOCH EVALUATION
                if is_epoch_evaluation and self.best_epoch_based_val_cl_loss > epoch_val_cl_loss:
                    print(f'    After epoch evaluation: Previous validation contrastive loss {self.best_epoch_based_val_cl_loss} --> New validation contrastive loss {epoch_val_cl_loss}')
                    self.best_epoch_based_val_cl_loss = epoch_val_cl_loss
                    self.early_stop_counter = 0 # reset if there are any improvement
                    self._save_ckpt(epoch, 
                                    'CTClip_lowest_val_cl_loss_after_per_epochs.pt', 
                                    'best contrastive loss on validation split!!', 
                                    iteration)
                elif is_epoch_evaluation: # implies that based on epoch-to-epoch comparison, there is not improvement
                    # early stopping based on val cl loss
                    self.early_stop_counter += 1
                    print(f"No improvement in validation loss for {self.early_stop_counter} epochs.")
                    if self.early_stop_counter >= self.epoch_based_patience:
                        print("Early stopping triggered. Stopping training.")
                        return True # Exit training loop

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