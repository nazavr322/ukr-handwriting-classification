from typing import ClassVar
import torch.nn as nn


class HandwritingClassifier(nn.Module):
    """ A simple architecture for handwritten token classification """
    
    # model expects normalized input using following mean and std
    _mean: ClassVar[tuple[float, ...]] = (0.1307, 0.1307, 0.1307)
    _std: ClassVar[tuple[float, ...]] = (0.3081, 0.3081, 0.3081)

    def __init__(self):
        super().__init__()
        self.conv_stack_1 = nn.Sequential(            
            nn.Conv2d(3, 32, 3, padding=1),  # batch_size x 32 x 28 x 28
            nn.MaxPool2d(2),  # batch_size x 32 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.conv_stack_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),  # batch_size x 64 x 14 x 14
            nn.MaxPool2d(2),  # batch_size x 64 x 7 x 7
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.flattener = nn.Flatten()  # batch_size x 3136 (64 * 7 * 7)
        self.fc_stack = nn.Sequential(
            nn.Linear(3136, 256),  # batch_size x 256
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        self.class_fc = nn.Linear(256, 10)  # batch_size x 10

    def forward(self, x):
        x = self.conv_stack_1(x)
        x = self.conv_stack_2(x)
        x = self.flattener(x)
        x = self.fc_stack(x)
        logits = self.class_fc(x)
        return logits
