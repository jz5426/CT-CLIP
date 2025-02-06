import timm
from timm.models import create_model
import torch

if __name__ == '__main__':
    checkpoint_path = '/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/models/vim_s_midclstok_ft_81p6acc.pth'
    models = timm.list_models('*res*')
    model = create_model(
        'vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2',
        pretrained=False,
        num_classes=18,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=448,
        dual_mode=False, # default choice
        cat_cls=False, # concatenate classification tokens?
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # code from main.py in BI-MAMBA REPO
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    if 'pos_embed' in checkpoint_model:
        print(f"Removing pos_embed from pretrained checkpoint")
        del checkpoint_model['pos_embed']

    missing, unexpected = model.load_state_dict(checkpoint_model, strict=False)
    print('model loaded')