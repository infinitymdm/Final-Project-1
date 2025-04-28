#! /usr/bin/env python3

import torchaudio.pipelines as pipelines
import torch.nn as nn

class FillerDetector(nn.Module):
    def __init__(self, bundle=pipelines.WAV2VEC2_BASE, out_dim=2, train_transfer_model=False, hidden_dim=256):
        super().__init__()
        self.transfer_model = bundle.get_model()
        self.train_transfer_model(train_transfer_model)
        classifier_in_dim = bundle._params['encoder_embed_dim']
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, out_dim)
        )
        #self.classifier = nn.Linear(classifier_in_dim, out_dim)

    def train_transfer_model(self, enable):
        '''Whether to train the parameters of the transfer model.

        Disable to train only the final layer'''
        for param in self.transfer_model.parameters():
            param.requires_grad = enable

    def forward(self, x):
        features,_ = self.transfer_model(x)
        pooled = features.mean(dim=1) # This is just global average pooling, it allows for different number of time steps if needed
        return self.classifier(pooled).squeeze(1) # This makes sure the output is not multidimensional

if __name__ == "__main__":
    model = FillerDetector()
    print(model)
