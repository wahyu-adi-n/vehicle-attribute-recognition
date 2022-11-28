import torch
import backbones

from utils.fsl_utils import compute_prototypes


class PrototypicalNetworks(torch.nn.Module):
    def __init__(self, backbone_name, variant_depth, dropout):
        super(PrototypicalNetworks, self).__init__()
        if backbone_name == 'effinet':
            self.backbone = backbones.effinet(
                variant=variant_depth,
                dropout=float(dropout)
            )
        elif backbone_name == 'resnet':
            self.backbone = backbones.resnet(
                variant=variant_depth,
                dropout=float(dropout)
            )
        elif backbone_name == 'convnet':
            self.backbone = backbones.ConvNet(
                depth=int(variant_depth),
                dropout=float(dropout)
            )
        else:
            ValueError(
                '{} as the selected backbone was not found'.format(backbone_name))

    def forward(self,
                support_images,
                support_labels,
                query_images):
        """
        Predict query labels using labeled support images.
        """
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)
        z_proto = compute_prototypes(z_support, support_labels)
        dists = torch.cdist(z_query, z_proto)
        scores = -dists
        return scores
