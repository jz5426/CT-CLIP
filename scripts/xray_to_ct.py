"""
TODO:
repeat the following and accumulate the stats:
    1. forward pass a synethic xray for each CT image in the validation set, to the pretrained xray encoder from ULIP style training
    2. with the embedding from xray encoder, compare it with the existing embeddings of the CT images based the image_embeddings.pth and find the top-k accuracy
        - note that since we forward pass the syntheic xray of examing ct image, we know the ground truth based on the dictionary id from image_embeddings.
"""