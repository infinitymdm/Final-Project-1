import torchaudio.pipelines as pipelines
import torch.nn as nn

class Filler_Detector(nn.Module):
    def __init__(self, transfer_model, in_dim, out_dim=2):
        super(Filler_Detector, self).__init__()
        self.transfer_model = transfer_model
        self.classifier = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        features,_ = self.transfer_model(x)
        pooled = features.mean(dim=1) # This is just global average pooling, it allows for different number of time steps if needed
        return self.classifier(pooled).squeeze(1) # This makes sure the output is not multidimensional

if __name__ == "__main__":
    ONLY_FINAL_LAYER_TRAINING = False

    bundle = pipelines.WAV2VEC2_BASE
    model = bundle.get_model()

    #print(model)
    #print(bundle._params['encoder_embed_dim'])

    FINAL_LAYER_IN = bundle._params['encoder_embed_dim']

    if ONLY_FINAL_LAYER_TRAINING:
        for params in model.parameters():
            params.requires_grad = False

    fd_model = Filler_Detector(model, FINAL_LAYER_IN)
    print(fd_model)