import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image

import matplotlib.pyplot as plt


def imshow(tensor):
    image = tensor.clone().detach()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)


def load_image(image_path, shape=None):
    image = Image.open(image_path)
    image = transforms.Resize((shape, shape))(image)
    image = transforms.ToTensor()(image).unsqueeze(0)
    return image


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # Target neuron activations for the content image
        self.target_vects = {i: val.detach() for i, val in enumerate(target)}
        self._weight = 1
        self.layers = [5]
        self.cntr = 1

    def forward(self, x):
        loss = []
        for i in self.layers:
            layer_loss = F.mse_loss(x[i], self.target_vects[i])
            _loss = layer_loss
            loss.append(_loss)

        self.loss = sum(loss) * self._weight
        return self.loss


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        # Create a target correlation matrix for each output layer
        self.target_vects = {
            i: self.gram_matrix(val).detach() for i, val in enumerate(target_feature)
        }
        # Style Loss Weights (layerwise)
        self._weight = 1e9
        self.layers = [0, 1, 2, 3, 4]
        self.layer_weights = [10, 5, 1, 1, 1]

    def forward(self, x):
        loss = []
        for i in self.layers:
            G = self.gram_matrix(x[i])
            layer_loss = F.mse_loss(G, self.target_vects[i])
            _loss = layer_loss * self.layer_weights[i]
            loss.append(_loss)
        self.loss = sum(loss) * self._weight
        return self.loss

    def gram_matrix(self, x):
        (b, c, h, w) = x.size()
        features = x.view(b * c, h * w)  # convert the matrix shape to (c, h*W)
        G = torch.mm(features, features.t())
        return G.div(b * c * h * w)


class VGG(nn.Module):
    def __init__(self):
        """
        Modify the VGG-19 model bu slicing the feature extraction layers,
        freezing model weights and replacing the MaxPool2d with AvgPool2d
        """
        super(VGG, self).__init__()

        model = models.vgg19(pretrained=True).features[:35]
        for i, layer in model._modules.items():
            if type(layer) == nn.modules.activation.ReLU:
                #  DO NOT modify the input directly
                model._modules[i] = nn.modules.activation.ReLU(inplace=False)
            elif type(layer) == torch.nn.modules.pooling.MaxPool2d:
                # Change MaxPool2d layer to AvgPool2d for smoother transition as indicated in the paper
                model._modules[i] = nn.AvgPool2d(
                    kernel_size=2, stride=2, padding=0, ceil_mode=False
                )
        self.features = model.eval()

    def forward(self, x):
        # Capture and output the inactivated ConvNet neurons from identified layers
        features = []
        for layer_num, layer in enumerate(self.features):
            x = layer(x)
            if layer_num in {2, 5, 10, 19, 28, 30}:
                features.append(x)
        return features
