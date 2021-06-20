import torch
import segmentation_models_pytorch as smp


def DeepLabV3Plus(encoder_name='senet154',
                encoder_weights='imagenet',
                in_channels=3,
                classes=2):

    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=classes,                      # model output channels (number of classes in your dataset)
    )
