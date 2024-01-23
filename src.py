import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from utils import VGG, ContentLoss, StyleLoss
from utils import load_image, imshow


# Define a function to perform style transfer
def style_transfer(content_path, style_path, num_steps=8):
    content = load_image(content_path, shape=512).to(device)
    style = load_image(style_path, shape=512).to(device)

    # Referance image to apply style transfer on
    input_img = content.clone()

    # Load the feature extractor model and compute the target vectors
    model = VGG().to(device).eval()
    content_features = model(input_img)
    content_losses = ContentLoss(content_features)

    style_features = model(style)
    style_losses = StyleLoss(style_features)

    # We used LBFGS as the optimizer but Adam also works well however the convergence is much slower.
    # Optimize the input_img parameters !!!
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    for epoch in range(num_steps):

        def closure():
            global idx_epoc
            optimizer.zero_grad()
            _features = model(input_img)

            content_score = content_losses(_features)
            style_score = style_losses(_features)

            loss = content_score + style_score
            loss.backward(retain_graph=True)

            input_img.data.clamp_(0, 1)

            print(f"{content_score=}, {style_score=}")
            writer.add_scalar("Content Loss", content_score, idx_epoc)
            writer.add_scalar("Style Loss", style_score, idx_epoc)

            idx_epoc += 1
            return loss

        optimizer.step(closure)
        print(epoch)

    writer.close()
    return input_img


if __name__ == "__main__":
    # Perform style transfer on device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If need to plot the learning curves
    writer = SummaryWriter()

    # Load images
    content_img_path = "/Users/kaangun/repos/ECSE526/A4/other/content.jpg"
    style_img_path = "/Users/kaangun/repos/ECSE526/A4/other/style.jpg"

    idx_epoc = 0
    output = style_transfer(content_img_path, style_img_path)

    # Display the result
    imshow(output)
    plt.show()
