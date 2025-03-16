import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class LinearProbeModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        """
        this model is used for during lp training
            - this model is used with pre-extracted xray latents

        XrayClassificationModel is used for testing and it need full forward pass of the xray encoder + latent projection + classifier
            - the classifier layer should be the one trained with this class
        """
        super(LinearProbeModel, self).__init__()
        # NOTE: the linear layer should be pretrained
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # NOTE: assume x is normalized
        return self.fc(x)

class XrayClassificationModel(nn.Module):
    def __init__(self, 
                vision_model: nn.Module, 
                feature_projector: nn.Module, 
                pretrained_classifier: nn.Module = None,
                vision_model_type=''):
        """
        Args:
            vision_model (nn.Module): Pretrained vision model.
            isLinearProbe (bool): If True, freeze the weights of the vision model.
            in_features (int): Number of input features for the fully connected layer.
            num_classes (int): Number of output classes for the fully connected layer.
        """
        super(XrayClassificationModel, self).__init__()
        
        # Assign the vision model
        self.vision_model = vision_model
        self.vision_model_type = vision_model_type

        # Add a fully connected layer
        self.to_xray_latent=feature_projector
        
        # Freeze the vision model and the projection layer because this is linear probing
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.to_xray_latent.parameters():
            param.requires_grad = False

        # note that this probing layer is deliberately in additional to the to_xray_latent during pretraining.
        self.fc = pretrained_classifier
        for param in self.fc.parameters():
            param.requires_grad = False
        print('Pretrained classifier is loaded.')

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Forward through the vision model
        enc_xray = self.vision_model(x) # [8, 49, 768]

        if 'resnet' in self.vision_model_type.lower() or 'densenet' in self.vision_model_type.lower():
            enc_xray = enc_xray.view(enc_xray.shape[0], 1, -1)
        elif self.vision_model_type.lower() == 'medclip_vit':
            enc_xray = enc_xray.last_hidden_state
        elif 'bi-mamba' in self.vision_model_type.lower():
            # bi-mamaba already has meaned before the output so we can skip some steps
            xray_latents = self.to_xray_latent(enc_xray)
            xray_latents = F.normalize(xray_latents)
            output = self.fc(xray_latents)
            return output 

        enc_xray = torch.mean(enc_xray, dim=1) # pool the patch features # [8, 768]
        enc_xray = enc_xray.view(enc_xray.shape[0], -1) # global view for each xray in a batch
        xray_embeds = enc_xray[:, :] if enc_xray.ndim == 3 else enc_xray

        # projection and normalize the features, exactly the way during pretraining
        xray_latents = self.to_xray_latent(xray_embeds) # [8, 512] # NOTE: assume this is pretrained.
        xray_latents = F.normalize(xray_latents, dim = -1) # NOTE: IMPORTANT!

        # Forward through the fully connected layer and output logits
        # this is the extra layer to learn
        output = self.fc(xray_latents)
        
        return output

def proportion_mapping(proportion):
    if str(proportion) == '0.01':
        return 'one_percent'
    if str(proportion) == '0.025':
        return 'two_five_percent'
    if str(proportion) == '0.05':
        return 'five_percent'
    if str(proportion) == '0.1':
        return 'ten_percent'
    if str(proportion) == '1' or str(proportion) == '1.':
        return 'hundred_percent'
    
def  get_clean_model_name(messy_custom_model_name):
    # 'modeltype_Resnet__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch',

    messy_custom_model_name=messy_custom_model_name.lower()
    if 'batchstyle' in messy_custom_model_name: # identifier for custom pretrained model: whether the file name has cxr_clip or not
        parts = []
        if 'resnet' in messy_custom_model_name:
            parts.append('resnet')
        elif 'swin' in messy_custom_model_name:
            parts.append('swin')
        elif 'mamba' in messy_custom_model_name:
            parts.append('mamba')
        
        if 'pretrained_true' in messy_custom_model_name:
            parts.append('pretrained')
        
        if 'experiment' in messy_custom_model_name:
            parts.append('exp')
        elif 'patient' in messy_custom_model_name:
            parts.append('pat')
        elif 'instance' in messy_custom_model_name:
            parts.append('ins')

        if 'siamese' in messy_custom_model_name:
            parts.append('siamese')
        else:
            parts.append('infoNCE') # default option (even the model name does not have this)

        if 'textcl_0__ctcl_1' in messy_custom_model_name:
            parts.append('textcl_0__ctcl_1')
        elif 'textcl_1__ctcl_0' in messy_custom_model_name:
            parts.append('textcl_1__ctcl_0')

        return '_'.join(parts)
    else:
        return messy_custom_model_name


def get_cxr_clip_variants():
    cxr_clip_variants = ['cxr_clip_swin_m', 'cxr_clip_swin_mc', 'cxr_clip_swin', 'cxr_clip_resnet_m', 'cxr_clip_resnet_mc', 'cxr_clip_resnet']
    return cxr_clip_variants

def metadata_base_on_model_type(baseline_type, pth_trailing_string='features'):
    
    cxr_clip_variants = get_cxr_clip_variants()
    # assert baseline_type in ['cxr_clip_resnet', 'cxr_clip_swin', 'medclip_resnet', 'medclip_vit', 'gloria_densenet', 'gloria_resnet']
    if baseline_type in cxr_clip_variants: # can be either cxr_clip_swin or cxr_clip_resnet
        xray_model_type = baseline_type #'cxr_clip_swin' if cfg['model']['image_encoder']['model_type'] == 'swin' else 'cxr_clip_resnet'
        dim_xray = 768 if 'swin' in baseline_type else 2048  # if cfg['model']['image_encoder']['model_type'] == 'swin' else 2048

        # different pretrained variant of cxr_clip backbones
        if 'cxr_clip_swin_m' == baseline_type:
            model = 'swinM'
        elif 'cxr_clip_swin_mc' ==  baseline_type:
            model = 'swinMC'
        elif 'cxr_clip_swin' == baseline_type:
            model = 'swin'
        elif 'cxr_clip_resnet_m' == baseline_type:
            model = 'resnetM'
        elif 'cxr_clip_resnet_mc' == baseline_type:
            model = 'resnetMC'
        elif 'cxr_clip_resnet' == baseline_type:
            model = 'resnet'
        else:
            assert False

        pth_base_name = f'{model}_cxr_xray_{pth_trailing_string}.pth' if 'swin' in xray_model_type else f'{model}_cxr_xray_{pth_trailing_string}.pth'
        latent_size = 512
    elif baseline_type == 'medclip_resnet':
        xray_model_type = baseline_type
        dim_xray = 2048
        pth_base_name = f'resnet_medclip_{pth_trailing_string}.pth'
        latent_size = 512
        # place this somewhere in the medclip code to remove the learnt fc connected layer at the end, just like cxr_clip: del self.resnet.fc
    elif baseline_type == 'medclip_vit':
        xray_model_type = baseline_type
        dim_xray = 768
        pth_base_name = f'swin_medclip_{pth_trailing_string}.pth'
        latent_size = 512
    elif baseline_type == 'gloria_densenet':
        xray_model_type = baseline_type
        dim_xray = 1024
        pth_base_name = f'densenet_gloria_{pth_trailing_string}.pth'
        latent_size = 768
    elif baseline_type == 'gloria_resnet':
        xray_model_type = baseline_type
        dim_xray = 2048
        pth_base_name = f'resnet_gloria_{pth_trailing_string}.pth'
        latent_size = 768 # the final size of the xray embedding is indeed different in gloria
    elif baseline_type == 'bi-mamba':
        xray_model_type = baseline_type
        dim_xray = 1000
        pth_base_name = f'bi_mamba_{pth_trailing_string}.pth'
        latent_size = 1000 # no projection layer => the same as the dim_xray
    elif baseline_type == 'medklip_resnet':
        xray_model_type = baseline_type
        dim_xray = 256
        pth_base_name = f'medklip_resnet_{pth_trailing_string}.pth'
        latent_size = 256 # no projection layer => the same as the dim_xray
    else:
        xray_model_type = baseline_type
        dim_xray = 768 if 'swin' in baseline_type.lower() else 2048
        pth_base_name = f'{xray_model_type}_xray_{pth_trailing_string}.pth'
        latent_size = 512

    return dim_xray, xray_model_type, pth_base_name, latent_size

def save_metric_results(path, file_name, df, override_previous_saved_results=False):
    assert 'csv' in file_name

    csv_filename = os.path.join(path, file_name)
    file_exists = os.path.isfile(csv_filename)
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    
    # append the stats to the existing metric results (if exists)
    if override_previous_saved_results:
        df.to_csv(csv_filename, mode='w', index=False, header=True)
        print(f"New file created: {csv_filename}")
    else: # append if there exist a file.
        df.to_csv(csv_filename, mode='a', index=False, header=not file_exists)
        print(f"Data appended to {csv_filename}" if file_exists else f"New file created: {csv_filename}")
