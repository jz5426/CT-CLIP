import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearProbeModel(nn.Module):
    def __init__(self, linear_layer: nn.Module):
        """
        this model is used for during lp training
            - this model is used with pre-extracted xray latents

        XrayClassificationModel is used for testing and it need full forward pass of the xray encoder + latent projection + classifier
            - the classifier layer should be the one trained with this class
        """
        super(LinearProbeModel, self).__init__()
        # NOTE: the linear layer should be pretrained
        self.classification_layer = linear_layer

    def forward(self, x):
        return self.classification_layer(x)

class XrayClassificationModel(nn.Module):
    def __init__(self, vision_model: nn.Module, feature_projector: nn.Module, in_features: int, num_classes: int, pretrained_classifier: nn.Module = None):
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
        # Add a fully connected layer
        self.to_xray_latent=feature_projector
        
        # Freeze the vision model and the projection layer because this is linear probing
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.to_xray_latent.parameters():
            param.requires_grad = False

        # note that this probing layer is deliberately in additional to the to_xray_latent during pretraining.
        if pretrained_classifier:
            self.fc = pretrained_classifier
            print('Pretrained classifier is loaded.')
        else:
            self.fc = nn.Linear(in_features, num_classes)

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
        enc_xray = torch.mean(enc_xray, dim=1) # pool the patch features # [8, 768]
        enc_xray = enc_xray.view(enc_xray.shape[0], -1) # global view for each xray in a batch
        xray_embeds = enc_xray[:, :] if enc_xray.ndim == 3 else enc_xray

        # projection and normalize the features, exactly the way during pretraining
        xray_latents = self.to_xray_latent(xray_embeds) # [8, 512] # NOTE: assume this is pretrained.
        xray_latents = F.normalize(xray_latents, dim = -1)

        # Forward through the fully connected layer and output logits
        # this is the extra layer to learn
        output = self.fc(xray_latents)
        
        return output

    def xray_feature_extraction(self):
        

# Example usage
# if __name__ == "__main__":
    # from torchvision.models import resnet18

    # # Load a pre-trained vision model
    # pretrained_model = resnet18(pretrained=True)

    # # Initialize the wrapper model
    # model = XrayClassificationModel(vision_model=pretrained_model, isLinearProbe=True, in_features=512, num_classes=10)

    # # Print model summary
    # print(model)