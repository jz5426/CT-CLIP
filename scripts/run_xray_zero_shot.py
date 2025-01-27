"""
TODO:
	- load the text encoder from the CT-CLIP (no freedom on this) (DONE)
	- load the vision encoder either the CT vision encoder or any Xray vision encoder (DONE)
	- load the dataset that the text+vision encoder supposed to be run on.
	- zero inference run function with metric evaluation
"""

import os
from cxr_clip_utils import convert_dictconfig_to_dict
from einops import rearrange
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from data import MimicCTReportXRayDataset
from data_inference import CTReportXRayDatasetinfer

from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from torch import einsum
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
import pandas as pd
from ct_clip import CTCLIPwithXray
import random
import numpy as np
import copy

from CTCLIPTrainer import UniqueLevelSampler

def cycle(dl):
    while True:
        for data in dl:
            yield data

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


@hydra.main(
	version_base=None,
	config_path="/cluster/home/t135419uhn/CT-CLIP/configs",
	config_name="train")
def main(cfg: DictConfig):

	OmegaConf.resolve(cfg)

	if "LOCAL_RANK" in os.environ:
		# for ddp
		# passed by torchrun or torch.distributed.launch
		local_rank = int(os.environ["LOCAL_RANK"])
	else:
		# for debugging
		local_rank = -1

	if local_rank < 1:
		print(f"Configurations:\n{OmegaConf.to_yaml(cfg)}")

	# seed_everything(1234)
	# torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True # efficient performance optimization.

	# seed everything
	seed = 1024
	random.seed(seed)    
	np.random.seed(seed)    
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)  # If using multiple GPUs

	run(cfg)


def run(cfg_dot):
	torch.cuda.empty_cache()

	# convert the config file to dictionary
	cfg = convert_dictconfig_to_dict(cfg_dot)

	#NOTE: you need to use the follownig command to copy and past to the location cp -rL /path/to/source_directory /path/to/destination_directory 
		# the copied files in the destination folder will behave like regular files and directories. You can copy and paste them as usual using a file manager

	# uhn cluster from local filesc
	#TODO: 
		# 1. copy the downloaded huggingface model in G:\Chris\CT-CLIP\predownloaded_models (shield external drive) to the CT-CLIP
		# 2. for the image_encoder section of the yaml file (such as clip_Swin_clincial), replace the directory to the correct one
	tokenizer = BertTokenizer.from_pretrained(
		'/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertTokenizer/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
		do_lower_case=True,
		local_files_only=True
	)
	text_encoder = BertModel.from_pretrained(
		'/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertModel/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
		local_files_only=True
	)

	image_encoder = CTViT(
		dim = 512,
		codebook_size = 8192,
		image_size = 480,
		patch_size = 20,
		temporal_patch_size = 10,
		spatial_depth = 4,
		temporal_depth = 4,
		dim_head = 32,
		heads = 8
	)

	# assert cfg_dot.zero_shot_params.baseline_type in ['cxr_clip_resnet', 'cxr_clip_swin', 'medclip_resnet', 'medclip_vit', 'gloria_densenet', 'gloria_resnet']
	if 'cxr_clip' in cfg_dot.zero_shot_params.baseline_type: # can be either cxr_clip_swin or cxr_clip_resnet
		xray_model_type = cfg_dot.zero_shot_params.baseline_type #'cxr_clip_swin' if cfg['model']['image_encoder']['model_type'] == 'swin' else 'cxr_clip_resnet'
		dim_xray = 768 if 'swin' in cfg_dot.zero_shot_params.baseline_type else 2048  # if cfg['model']['image_encoder']['model_type'] == 'swin' else 2048
		pth_base_name = 'swin_cxr_xray_features.pth' if 'swin' in xray_model_type else 'resnet_cxr_xray_features.pth'
	elif cfg_dot.zero_shot_params.baseline_type == 'medclip_resnet':
		xray_model_type = cfg_dot.zero_shot_params.baseline_type
		dim_xray = 2048
		pth_base_name = 'resnet_medclip_features.pth'
		# place this somewhere in the medclip code to remove the learnt fc connected layer at the end, just like cxr_clip: del self.resnet.fc
	elif cfg_dot.zero_shot_params.baseline_type == 'medclip_vit':
		xray_model_type = cfg_dot.zero_shot_params.baseline_type
		dim_xray = 768
		pth_base_name = 'swin_medclip_features.pth'
	elif cfg_dot.zero_shot_params.baseline_type == 'gloria_densenet':
		xray_model_type = cfg_dot.zero_shot_params.baseline_type
		dim_xray = 1024 #TODO: double check this.
		pth_base_name = 'densenet_gloria_features.pth'
	elif cfg_dot.zero_shot_params.baseline_type == 'gloria_resnet':
		xray_model_type = cfg_dot.zero_shot_params.baseline_type
		dim_xray = 2048
		pth_base_name = 'resnet_gloria_features.pth'
	elif cfg_dot.zero_shot_params.baseline_type == '': # default to cxr_clip_swin for placeholder
		xray_model_type = cfg_dot.zero_shot_params.baseline_type #'cxr_clip_swin' if cfg['model']['image_encoder']['model_type'] == 'swin' else 'cxr_clip_resnet'
		dim_xray = 768
		pth_base_name = 'swin_cxr_xray_features.pth'
	else:
		xray_model_type = cfg_dot.zero_shot_params.baseline_type
		dim_xray = 768 if 'swin' in cfg_dot.zero_shot_params.baseline_type.lower() else 2048
		pth_base_name = f'{xray_model_type}_xray_features.pth'

	# load the xray encoder or the pretrained one depends on the xray_model_type
	# NOTE: this automatically loaded the xray encoderd depends on the baseline or type
	clip_xray = CTCLIPwithXray(
		image_encoder = image_encoder,
		text_encoder = text_encoder,
		# tokenizer=tokenizer,
		dim_text = 768,
		dim_image = 294912,
		xray_model_type = xray_model_type,
		dim_xray = dim_xray,
		dim_latent = 512,
		extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
		use_mlm=False,
		downsample_image_embeds = False,
		use_all_token_embeds = False,
		cfg=cfg,
		auto_load_pretrained_weights=True # because it loads it later.
	)
	clip_xray.load_ctclip('/cluster/home/t135419uhn/CT-CLIP/models/CT-CLIP_v2.pt')
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# pick the text encoder and its associated latent projector
	report_encoder = copy.deepcopy(clip_xray.CTCLIP.text_transformer)
	report_latent_projecter = copy.deepcopy(clip_xray.CTCLIP.to_text_latent)
	report_encoder.to(device)
	report_latent_projecter.to(device)
	report_encoder.eval()
	report_latent_projecter.eval()

	# pick the vision encoder (ct_encoder or any xray encoder, depends on the config params) and its associated latent projector
	vision_encoder = copy.deepcopy(clip_xray.CTCLIP.visual_transformer if cfg_dot.zero_shot_params.vision_encoder == 'ct_clip' else clip_xray.xray_encoder)
	vision_latent_projector = copy.deepcopy(clip_xray.CTCLIP.to_visual_latent if cfg_dot.zero_shot_params.vision_encoder == 'ct_clip' else clip_xray.to_xray_latent)
	vision_encoder.to(device)
	vision_latent_projector.to(device)
	vision_encoder.eval()
	vision_latent_projector.eval()

	# load the test bed (the dataset) for zero-shot experiment
	# e.g: the internal val dataset, the mimic dataset, and potential future integration
	
	# some integrity constraint
	if cfg_dot.zero_shot_params.vision_encoder == 'ct_clip':
		assert cfg_dot.zero_shot_params.test_bed == 'internal_ct_val'
	if cfg_dot.zero_shot_params.vision_encoder != 'ct_clip':
		assert cfg_dot.zero_shot_params.test_bed != 'internal_ct_val'

	# load the dataset for test zero-shot performance.
	if cfg_dot.zero_shot_params.test_bed == 'mimic_ct':
		test_bed = MimicCTReportXRayDataset(
			cfg=cfg,
			data_folder='/cluster/home/t135419uhn/CT-CLIP/preprocessed_mimic/mimic_preprocessed_xray_mha',
			csv_file='/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/external_valid_mimic_report.csv',
			labels='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_external_valid_mimic_labels.csv', 
			split='valid' # for transformation
		)
		# xray_image, report, label, accession_number
		print(f'size of the {cfg_dot.zero_shot_params.test_bed}: {len(test_bed)}')
		pathologies = ['Arterial wall calcification',
						'Pericardial effusion',
						'Coronary artery wall calcification',
						'Hiatal hernia',
						'Lymphadenopathy',
						'Emphysema',
						'Atelectasis',
						'Mosaic attenuation pattern',
						'Peribronchial thickening',
						'Bronchiectasis',
						'Interlobular septal thickening'
		]
	elif cfg_dot.zero_shot_params.test_bed == 'internal_ct_val':
		split = 'valid'
		#NOTE: this returns the ct embeddings and the text embeddings and the xray image (paired up)
		test_bed = CTReportXRayDatasetinfer(
			data_folder=f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/{split}_preprocessed_xray_mha', # THIS IS CORRECT
			cfg=cfg, 
			csv_file=f'/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/{split}_reports.csv',
			img_embedding_path=f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/features_embeddings/{split}/image_features.pth',
			text_embedding_path=f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/features_embeddings/{split}/text_features.pth',
			batch_style='instance',
			labels=f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv'
		)
		# img_embedding, text_embedding, onehotlabels, xray_image, name_acc, xray_file
		print(f'size of the {cfg_dot.zero_shot_params.test_bed}: {len(test_bed)}')
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
						'Interlobular septal thickening'
		]

        # custom_sampler = UniqueLevelSampler(test_bed.key_ids, cfg_dot.zero_shot_params.batch_size)
		custom_sampler = UniqueLevelSampler(test_bed.key_ids, cfg_dot.zero_shot_params.batch_size)
		test_bed_dl = DataLoader(
			test_bed,
			num_workers=cfg_dot.zero_shot_params.num_workers,
			shuffle = False,
			batch_sampler=custom_sampler
		)
		test_bed_iter = cycle(test_bed_dl)
	else:
		NotImplementedError(f'{cfg_dot.zero_shot_params.test_bed} is not supported at the moment')
	
	# run zero-shot evaluation
	zero_shot_evaluation(
		test_bed_iter,
		cfg_dot,
		pathologies,
		tokenizer,
		report_encoder,
		report_latent_projecter
	)

	#NOTE: directly compares the embeddings for zero-shot evaluation but not other dataset

def zero_shot_evaluation(valid_dl_iter, cfg, pathologies, tokenizer, report_encoder, report_latent_projector):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	all_labels, all_preds, all_probs  = [], [], []

	with torch.no_grad():
		for val_data in valid_dl_iter: #NOTE: might need to change this to evaluate on the whole validation set.

			if cfg.zero_shot_params.test_bed == 'internal_ct_val':
				ct_latents, text_latents, onehotlabels, xray_image, _, _ = val_data
				text_latents = text_latents.to(device)
				ct_latents = ct_latents.to(device)

			elif cfg.zero_shot_params.test_bed == 'mimic_ct':
				# valid_data is the xray_image
				xray_image, report, onehotlabels, _ = val_data
				xray_image = xray_image.to(device)

			# make zero-shot prediction for each batch of data of the test bed by computing dot product between text with ct or text and xray
			for pathology in pathologies:
				text = [f"There is {pathology}.", f"There is no {pathology}."] #NOTE: binary classification for each pathology.
				text_tokens=tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

				if cfg.zero_shot_params.test_bed == 'internal_ct_val':
					# similarity between text and ct , the text should be the 
					num_batch_texts = num_batch_images = 1
					text_latents = rearrange(text_latents, '(m b) ... -> m b ...', m = num_batch_texts) #NOTE: 1xbxd
					ct_latents = rearrange(ct_latents, '(m b) ... -> m b ...', m = num_batch_images) #NOTE: 1xbxd
					logits = einsum('m t d, n i d -> m n t i', text_latents, ct_latents).squeeze() # [size of the ct, size of the text]
				elif cfg.zero_shot_params.test_bed == 'mimic_ct':
					pass

				# TODO: MAKE SURE TO NORMALIZE THE FEATURE BEFORE PERFORMING COMPARISON
			
				probs = torch.sigmoid(logits)
				preds = (probs > 0.5).int()
				all_labels.extend(onehotlabels.cpu().numpy())
				all_preds.extend(preds.cpu().numpy())
				all_probs.extend(probs.cpu().numpy())

		# Convert to numpy arrays for metric computation
		all_labels = np.array(all_labels)
		all_preds = np.array(all_preds)
		all_probs = np.array(all_probs)

		# Calculate metrics for multilabel classification
		# NOTE: might use the same one from the training file instead of using the sklearn one.
		precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')
		# precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
		# precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

		auc_micro = roc_auc_score(all_labels, all_probs, average='micro', multi_class='ovr')
		# auc_weighted = roc_auc_score(all_labels, all_probs, average='weighted', multi_class='ovr')
		# auc_macro = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
		pr_auc_score = average_precision_score(all_labels, all_probs, average='micro')

		print(f'Test results for micro average: PR_AUC: {pr_auc_score:.4f}')
		print(f"Test Results for micro average: Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1 Score: {f1_micro:.4f}, AUC: {auc_micro:.4f}")
		# print(f"Test Results for weighted average: Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}, F1 Score: {f1_weighted:.4f}, AUC: {auc_weighted:.4f}")
		# print(f"Test Results for macro average: Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1 Score: {f1_macro:.4f}, AUC: {auc_macro:.4f}")

		print('Saving the metrics results')
		metrics_data = {
			'Metric': ['Precision', 'Recall', 'F1 Score', 'AUC', 'PR_AUC'],
			'Micro': [precision_micro, recall_micro, f1_micro, auc_micro, pr_auc_score],
			# 'Weighted': [precision_weighted, recall_weighted, f1_weighted, auc_weighted, -1],
			# 'Macro': [precision_macro, recall_macro, f1_macro, auc_macro, -1]
		}

		metrics_df = pd.DataFrame(metrics_data)
		os.makedirs(os.path.dirname(metric_saving_path), exist_ok=True)
		metrics_df.to_excel(metric_saving_path, index=False)
		print(f"Metric results saved to {metric_saving_path}")



if __name__ == '__main__':
	main()