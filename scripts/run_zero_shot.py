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
# clip.load("C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\chris\\CT-CLIP\\models\\CT-CLIP_v2.pt")

# '/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/valid/'
# inference = CTClipInference(
#     clip,
#     data_folder = "F:\\Chris\\dataset\\valid\\valid_preprocessed_ct",
#     reports_file= "C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\chris\\CT-CLIP\\dataset\\radiology_text_reports\\valid_reports.csv",
#     labels = "C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\chris\\CT-CLIP\\dataset\\multi_abnormality_labels\\dataset_multi_abnormality_labels_valid_predicted_labels.csv",
#     batch_size = 1,
#     results_folder="inference_zeroshot\\",
#     num_train_steps = 1,
#     feature_extraction_mode = True # extract only the text and ct features only
# )
# inference.infer()

# NOTE: to work on the WSL drive, do the sudo mount -t drvfs F: /mnt/f if the external drive has no content
# inference_valid = CTClipInference(
#     clip,
#     data_folder = "/mnt/f/Chris/dataset/valid_preprocessed_ct",
#     reports_file= "/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/radiology_text_reports/valid_reports.csv",
#     labels = "/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv",
#     batch_size = 1,
#     results_folder="inference_zeroshot/",
#     num_train_steps = 1,
#     feature_extraction_mode = True # extract only the text and ct features only
# )

# # inference_valid.infer()
# inference_valid.feature_extraction('/mnt/f/Chris/dataset/features_embeddings', 'valid')

inference_train = CTClipInference(
    clip,
    data_folder = "/mnt/f/Chris/CT-RATE-temp/processed_dataset/train_preprocessed_ct", # "/mnt/f/Chris/dataset/train_preprocessed_ct",
    reports_file= "/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/radiology_text_reports/train_reports.csv",
    labels = "/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_predicted_labels.csv",
    batch_size = 4,
    results_folder="inference_zeroshot/",
    num_train_steps = 1,
    feature_extraction_mode = True # extract only the text and ct features only
)

# inference_train.infer()
inference_train.feature_extraction('/mnt/f/Chris/dataset/features_embeddings', 'train')

"""
TODO: update the .ph object if exists (DONE)
TODO: train in epochs instead of iterations (DONE)
TODO: change the download script such that it is in the unit of patient instead of number of volumes
        - which also should take into account the cases where the the volumes (via a path) already exists
        - more efficiently, just write a a script that download the data and process it, and then save only the embeddings and the corresponding x-rays (TODO:)
TODO: hyperparameters for the xray encoder with temperature 0.07
TODO: batch based on patient/experiment/instance.
"""
