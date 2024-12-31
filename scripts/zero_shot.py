from pathlib import Path
from shutil import rmtree
from transformer_maskgit.optimizer import get_optimizer
from transformers import BertTokenizer, BertModel

from data import CTReportDataset, CTReportXRayDataset
from eval import evaluate_internal, plot_roc, accuracy, sigmoid, bootstrap, compute_cis

from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

# from data_inference_nii import CTReportDatasetinfer
# from data_external_valid import CTReportDatasetinfer
from data_inference import CTReportDatasetinfer, CTReportXRayDatasetinfer

import numpy as np
import tqdm
import pandas as pd
import nibabel as nib

from einops import rearrange
import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import math
import torch.optim.lr_scheduler as lr_scheduler
from ct_clip import CTCLIP
import os

from scripts.CTCLIPTrainer import UniqueLevelSampler

# helpers

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

class CTClipInference(nn.Module):
    def __init__(
        self,
        CTClip: CTCLIP,
        *,
        num_train_steps,
        batch_size,
        cfg=None,
        num_workers = 1,
        img_embedding_paths = {},
        text_embedding_paths = {},
        feature_extraction_mode = False,
        data_folder = "external_valid",
        reports_file = "data_reports.xslx",
        lr = 1e-4,
        wd = 0.,
        max_grad_norm = 0.5,
        save_results_every = 100,
        save_model_every = 2000,
        results_folder = './results',
        labels = "labels.csv",
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)
        self.CTClip = CTClip
        self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
        self.results_folder = results_folder
        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        all_parameters = set(CTClip.parameters())

        self.optim = get_optimizer(all_parameters, lr=lr, wd=wd)

        self.max_grad_norm = max_grad_norm
        self.lr=lr

        # NOTE: automatic toggle: alter to ULIP-style mode if the xray encoder exists in the CTCLIP
        self.triplet = False
        if hasattr(self.CTClip, 'xray_encoder'):
            self.triplet = True
            # max_grad_norm = None # TODO: might need to experiment if need this.

        if self.triplet:
            assert(img_embedding_paths.keys() == text_embedding_paths.keys())
            assert(cfg is not None)

            # Load the pre-trained weights
            self.ds = CTReportXRayDataset(
                        data_folder=data_folder,
                        csv_file=reports_file,
                        cfg=cfg,
                        img_embedding_path=img_embedding_paths['train'], 
                        text_embedding_path=text_embedding_paths['train'],
                        batch_style='instance') if 'train' in img_embedding_paths else \
                      CTReportXRayDatasetinfer(
                        data_folder=data_folder, 
                        cfg=cfg, 
                        csv_file=reports_file,
                        img_embedding_path=img_embedding_paths['valid'],
                        text_embedding_path=text_embedding_paths['valid'],
                        batch_style='instance',
                        labels=labels)

            custom_sampler = UniqueLevelSampler(self.ds.key_ids, self.batch_size)
            self.dl = DataLoader(
                self.ds,
                num_workers=num_workers,
                batch_sampler=custom_sampler
            )

            self.split = 'valid' if 'valid' in img_embedding_paths else 'train'
        else:
            # Load the pre-trained weights
            self.ds = CTReportDatasetinfer(
                data_folder=data_folder,
                csv_file=reports_file,
                labels=labels)

            # Split dataset into train and validation sets
            self.dl = DataLoader(
                self.ds,
                num_workers=num_workers,
                batch_size=batch_size,
                shuffle = True,
            )

            self.split = 'valid' # default

        self.image_features = None
        self.text_features = None
        self.feature_extraction_mode = feature_extraction_mode
        if self.feature_extraction_mode:
            self.image_features = {}
            self.text_features = {}

        # prepare with accelerator
        self.dl_iter=cycle(self.dl)
        self.device = self.accelerator.device
        self.CTClip.to(self.device)
        self.lr_scheduler = CosineAnnealingWarmUpRestarts(self.optim,
                                                  T_0=4000000,    # Maximum number of iterations
                                                  T_warmup=10000, # Number of warmup steps NOTE: TODO: verify this 
                                                  eta_max=lr)   # Maximum learning rate


        (
 			self.dl_iter,
            self.CTClip,
            self.optim,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.CTClip,
            self.optim,
            self.lr_scheduler
        )

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.result_folder_txt = self.results_folder
        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(parents=True, exist_ok=True)

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

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        # logs
        logs = {}

        if True:
            with torch.no_grad():

                models_to_evaluate = ((self.CTClip, str(steps)),)

                for model, filename in models_to_evaluate:
                    model.eval()
                    predictedall=[]
                    realall=[]
                    logits = []

                    text_latent_list = []
                    image_latent_list = []
                    accession_names=[]
                    pathologies = ['Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening']
                    for i in tqdm.tqdm(range(len(self.ds))):
                        valid_data, text, onehotlabels, acc_name, instance_name, nii_file = next(self.dl_iter)

                        plotdir = self.result_folder_txt
                        Path(plotdir).mkdir(parents=True, exist_ok=True)

                        predictedlabels=[]
                        onehotlabels_append=[]
                        for pathology in pathologies:
                            text = [f"{pathology} is present.", f"{pathology} is not present."]
                            text_tokens=self.tokenizer(
                                            text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

                            output = model(text_tokens, valid_data.cuda(), device=device, return_latents=self.feature_extraction_mode)

                            output = apply_softmax(output)

                            append_out=output.detach().cpu().numpy()
                            predictedlabels.append(append_out[0])

                        predictedall.append(predictedlabels)
                        realall.append(onehotlabels.detach().cpu().numpy()[0])
                        accession_names.append(acc_name[0])

                    realall=np.array(realall)
                    predictedall=np.array(predictedall)

                    np.savez(f"{plotdir}labels_weights.npz", data=realall)
                    np.savez(f"{plotdir}predicted_weights.npz", data=predictedall)
                    with open(f"{plotdir}accessions.txt", "w") as file:
                        for item in accession_names:
                            file.write(item + "\n")

                    dfs=evaluate_internal(predictedall,realall,pathologies, plotdir)

                    writer = pd.ExcelWriter(f'{plotdir}aurocs.xlsx', engine='xlsxwriter')

                    dfs.to_excel(writer, sheet_name='Sheet1', index=False)

                    writer.close()
        self.steps += 1
        return logs

    def infer(self, log_fn=noop):
        device = next(self.CTClip.parameters()).device
        device=torch.device('cuda')
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('Inference complete')

    def ctclip_feature_extraction(self, directory, split='valid', append=True):
        # load the .pth object if exists
        saving_path = os.path.join(directory, split)
        img_feature_path = os.path.join(saving_path, 'image_features.pth')
        text_feature_path = os.path.join(saving_path, 'text_features.pth')
        if os.path.exists(img_feature_path):
            self.image_features = torch.load(img_feature_path)
        if os.path.exists(text_feature_path):
            self.text_features = torch.load(text_feature_path)

        assert(isinstance(self.image_features, dict))
        assert(isinstance(self.text_features, dict))

        device = self.device
        with torch.no_grad():
            self.CTClip.eval()
            # bar = tqdm.tqdm(self.dl, desc="Feature Extration", leave=False)
            idx = 0
            # for batch_data in tqdm.tqdm(self.dl, desc="Feature Extration", leave=False):
            #     valid_data, text, _, _, instance_name, _ = batch_data

            #     # batch processing
            #     text_tokens=self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
            #     output = self.CTClip(text_tokens, valid_data.cuda(), device=device, return_latents=self.feature_extraction_mode)
            #     text_feature, img_feature, _ = output
            #     text_feature, img_feature = text_feature.cpu().numpy(), img_feature.cpu().numpy()

            #     # assign the features inside the batch in the dict
            #     for i, key in enumerate(instance_name):
            #         self.image_features[key] = img_feature[i, :]
            #         self.text_features[key] = text_feature[i, :]
        
            #     # # save the feature embeddings every 100 iterations.
            #     if append and idx % 100 == 0:
            #         os.makedirs(saving_path, exist_ok=True)
            #         torch.save(self.image_features, img_feature_path)
            #         torch.save(self.text_features, text_feature_path)
            #     else:
            #         print('NOT SAVING IT THE EMBEDDINGS!!!')
            #     idx += 1

            for batch_data in tqdm.tqdm(self.dl, desc="Feature Extraction", leave=False):
                valid_data, text, _, _, instance_name, _ = batch_data

                # Filter out instance names that already exist in image_features and text_features
                new_instance_indices = [
                    i for i, key in enumerate(instance_name) 
                    if key not in self.image_features or key not in self.text_features
                ]
                if not new_instance_indices:
                    print("All keys in the batch already exist. Skipping model forward pass.")
                    continue  # Skip the current batch if all keys already exist

                # Select only the new data for processing
                valid_data = valid_data[new_instance_indices]
                text = [text[i] for i in new_instance_indices]
                instance_name = [instance_name[i] for i in new_instance_indices]

                # Tokenize and forward pass
                text_tokens = self.tokenizer(
                    text, return_tensors="pt", padding="max_length", truncation=True, max_length=512
                ).to(device)
                output = self.CTClip(
                    text_tokens,
                    valid_data.cuda(),
                    device=device,
                    return_latents=self.feature_extraction_mode,
                )
                text_feature, img_feature, _ = output
                text_feature, img_feature = text_feature.cpu().numpy(), img_feature.cpu().numpy()

                # Assign the features to the respective dictionaries
                for i, key in enumerate(instance_name):
                    self.image_features[key] = img_feature[i, :]
                    self.text_features[key] = text_feature[i, :]

                # Save the feature embeddings every 100 iterations
                if append and idx % 100 == 0:
                    os.makedirs(saving_path, exist_ok=True)
                    torch.save(self.image_features, img_feature_path)
                    torch.save(self.text_features, text_feature_path)
                else:
                    print("NOT SAVING THE EMBEDDINGS!!!")
                idx += 1

        
        # save the remaining.
        if append:
            os.makedirs(saving_path, exist_ok=True)
            torch.save(self.image_features, img_feature_path)
            torch.save(self.text_features, text_feature_path)
        else:
            print('NOT SAVING IT THE EMBEDDINGS!!!')

        #sanity check
        loaded_img_features = torch.load(os.path.join(saving_path, 'image_features.pth'))
        loaded_txt_features = torch.load(os.path.join(saving_path, 'text_features.pth'))
        print(f'size of image features {len(loaded_img_features)}; size of text features {len(loaded_txt_features)}')

    def xray_feature_extraction(self, directory, append=True):
        # sanity check
        assert(self.split in ['valid']) # NOTE: for train to work, need to change the __get_item__ method in the class to output the instance name
        assert(self.triplet == True)

        print('Retrieval Evaluation Starts\n')
        device = self.device
        data_size = len(self.valid_dl) if self.split == 'valid' else len(self.dl)
        data_iterator = self.valid_dl if self.split == 'valid' else self.dl

        # load the .pth object if exists
        saving_path = os.path.join(directory, self.split)
        xray_feature_path = os.path.join(saving_path, 'xray_features.pth')
        xray_features = {}
        if os.path.exists(xray_feature_path):
            xray_features = torch.load(xray_feature_path)

        with torch.no_grad():
            self.CTClip.eval()

            idx = 0
            # for batch_idx in range(data_size):
            for data in tqdm.tqdm(data_iterator, desc="XRay Feature Extraction", leave=False):
                # data = next(data_iterator)
                _, _, _, xray, instance_name, _ = data  # NOTE: double-check the instance name, depends on the custom data loader.

                # Filter out instance names that already exist in xray_features
                new_instance_indices = [i for i, key in enumerate(instance_name) if key not in xray_features]
                if not new_instance_indices:
                    print("All keys in the batch already exist. Skipping model forward pass.")
                    continue  # Skip the current batch if all keys already exist

                # Select only the new xray data for processing
                instance_name = [instance_name[i] for i in new_instance_indices]
                xray = xray[new_instance_indices].to(device)

                # Forward pass for the new xray latents
                batch_xray_latents = self.CTClip.get_xray_latents(xray)
                batch_xray_latents = batch_xray_latents.cpu().detach().numpy()

                # Assign the features inside the batch in the dict
                for i, key in enumerate(instance_name):
                    xray_features[key] = batch_xray_latents[i, :]

                # periodically save the features
                if append and idx % 100 == 0:
                    os.makedirs(saving_path, exist_ok=True)
                    torch.save(xray_features, xray_feature_path)
                else:
                    print('NOT SAVING IT THE EMBEDDINGS!!!')
                idx += 1

        if append:
            os.makedirs(saving_path, exist_ok=True)
            torch.save(xray_features, xray_feature_path)
        else:
            print('NOT SAVING IT THE EMBEDDINGS!!!')

        return xray_features