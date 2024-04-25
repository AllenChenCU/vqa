import math

import torch
import torch.nn as nn

import config


class SimpleNet(nn.Module):
    def __init__(self, pretrained_model):
        super(SimpleNet, self).__init__()
        self.pretrained_model = pretrained_model
        #self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, 1)
        v_size = 2048 * 14 * 14
        self.classifier = Classifier(
            in_features=self.pretrained_model.config.hidden_size + v_size, 
            mid_features=1024, 
            out_features=1, 
            drop=0.5, 
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, v: torch.Tensor, wrapped_text_input: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden_state = self.pretrained_model(**wrapped_text_input)
        token_hidden_state, cls_hidden_state = hidden_state[0], hidden_state[1]
        v = torch.flatten(v, start_dim=1)
        combined = torch.cat([v, cls_hidden_state], dim=1)
        logits = self.classifier(combined)
        probs = self.sigmoid(logits)
        return probs


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('fc1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('fc2', nn.Linear(mid_features, out_features))


class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
        pass

    def forward(self, v: torch.Tensor, wrapped_text_input: dict[str, torch.Tensor]) -> torch.Tensor:
        pass
