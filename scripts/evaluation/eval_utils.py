import torch
import torch.nn as nn

class XrayClassificationModel(nn.Module):
    def __init__(self, vision_model: nn.Module, isLinearProbe: bool, in_features: int, num_classes: int):
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
        
        # Freeze the vision model if isLinearProbe is True
        if isLinearProbe:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        # Add a fully connected layer
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
        features = self.vision_model(x)

        # Forward through the fully connected layer and output logits
        output = self.fc(features)
        
        return output

# Example usage
if __name__ == "__main__":
    # from torchvision.models import resnet18

    # # Load a pre-trained vision model
    # pretrained_model = resnet18(pretrained=True)

    # # Initialize the wrapper model
    # model = XrayClassificationModel(vision_model=pretrained_model, isLinearProbe=True, in_features=512, num_classes=10)

    # # Print model summary
    # print(model)
