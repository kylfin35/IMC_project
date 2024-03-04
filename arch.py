import numpy as np
import torch
import os 
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
from torchvision import transforms
import copy


class DINO(torch.nn.Module):
    def __init__(self, backbone, input_dim, freeze_vit = False, output_dims = 2048, normalize = False):
        super().__init__()
        self.freeze_vit = freeze_vit
        self.output_dims = output_dims
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, self.output_dims, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, self.output_dims)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)
        self.normalize = normalize
        if self.normalize:
          self.mean = self.normalize[0]
          self.std = self.normalize[1]
          self.norm_fn = transforms.Normalize(self.mean, self.std)

        if self.freeze_vit:
          for param in self.student_backbone.parameters():
            param.requires_grad = False
          for param in self.student_backbone.patch_embed.parameters():
            param.requires_grad = True
          for param in self.student_backbone.blocks[-2:].parameters():
            param.requires_grad = True
    def norm_input(self, x):
      x = self.norm_fn(x)
      x = torch.clamp(x, min=None, max=3.5)
      return x

    def forward(self, x):
        if self.normalize:
          x = self.norm_input(x)
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        if self.normalize:
            x = self.norm_input(x)
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z