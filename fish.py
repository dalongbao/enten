import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import Tuple

import math
import numpy as np

class Fish(nn.Module):
    def super().__init__(self):
        """
        start with a two layer MLP
        """
        self.model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
    
    def forward(self, x):
        return self.model(x)
