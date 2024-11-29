import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from zero_shot import CTClipInference
import accelerate

tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

text_encoder.resize_token_embeddings(len(tokenizer))


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

clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_image = 294912,
    dim_text = 768,
    dim_latent = 512,
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds = False,
    use_all_token_embeds = False

)

# clip.load("path_to_pretrained_model")

# inference = CTClipInference(
#     clip,
#     data_folder = 'path_to_preprocessed_validation_folder',
#     reports_file= "path_to_validation_reports_csv",
#     labels = "path_to_validation_labels_csv",
#     batch_size = 1,
#     results_folder="inference_zeroshot/",
#     num_train_steps = 1,
# )

clip.load("/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/models/CT-CLIP_v2.pt")

# 
# '/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/valid/'
inference = CTClipInference(
    clip,
    data_folder = '/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/valid_preprocessed_ct',
    reports_file= "/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/radiology_text_reports/dataset_radiology_text_reports_validation_reports.csv",
    labels = "/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv",
    batch_size = 1,
    results_folder="inference_zeroshot/",
    num_train_steps = 1,
)

inference.infer()
