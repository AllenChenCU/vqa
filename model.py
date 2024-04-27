import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class SimpleNet(nn.Module):
    def __init__(self, pretrained_model):
        super(SimpleNet, self).__init__()
        self.pretrained_model = pretrained_model
        self.glimpses = 2
        #self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, 1)

        self.attention = Attention(
            v_features=config.OUTPUT_FEATURES, 
            q_features=self.pretrained_model.config.hidden_size, 
            mid_features=512, 
            glimpses=self.glimpses, 
            drop=0.5, 
        )

        v_size = self.glimpses*config.OUTPUT_FEATURES
        #v_size = 2048 * 14 * 14
        q_size = self.pretrained_model.config.hidden_size
        self.classifier = Classifier(
            in_features=q_size + v_size, 
            mid_features=1024, 
            out_features=1, 
            drop=0.5, 
        )
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v: torch.Tensor, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_state = self.pretrained_model(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask, 
        )
        token_hidden_state, q = hidden_state[0], hidden_state[1] # q = cls hidden state

        # l2 normalization
        q = q / (q.norm(p=2, dim=1, keepdim=True).expand_as(q) + 1e-8)
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        w = self.attention(v, q) # (batch_size, glimpses, 14, 14)
        v = apply_attention(v, w) # (batch_size, 4096)
        #v = torch.flatten(v, start_dim=1)
        combined = torch.cat([v, q], dim=1)
        logits = self.classifier(combined)
        probs = self.sigmoid(logits)
        return probs


class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def tile_2d_over_nd(feature_vector, feature_map):
        """ Repeat the same feature vector over all spatial positions of a given feature map.
            The feature vector should have the same batch size and number of features as the feature map.
        """
        n, c = feature_vector.size()
        spatial_size = feature_map.dim() - 2
        tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
        return tiled

    def forward(self, v, q):
        v = self.v_conv(self.drop(v)) # (batch_size, mid_features, 14, 14)
        q = self.q_lin(self.drop(q)) # (batch_size, mid_features)
        q = Attention.tile_2d_over_nd(q, v)
        x = self.relu(v + q) # (batch_size, mid_features, 14, 14)
        x = self.x_conv(self.drop(x)) # (batch_size, glimpses, 14, 14)
        return x


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('fc1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('fc2', nn.Linear(mid_features, out_features))


def apply_attention(input, attention):
    """ Apply any number of attention maps over the input. 
    ex. input.shape = (batch_size, 2048, 14, 14)
        attention.shape = (batch_size, 2, 14, 14)
    """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, 1, c, -1) # [n, 1, c, s]

    attention = attention.view(n, glimpses, -1)
    attention = F.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s]

    weighted = attention * input # [n, g, v, s]
    weighted_mean = weighted.sum(dim=-1) # [n, g, v]
    return weighted_mean.view(n, -1) # [n, g*v]
