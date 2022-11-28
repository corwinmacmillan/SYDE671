import torch
import torch.nn as nn

class DestripeNet(nn.Module):
    '''
    DestripeNet model
    '''
    def __init__(
        self, 
        in_channels=38, 
        out_channels=1, 
        features=[512, 512, 512, 512, 512, 512, 512, 512, 256, 128, 64, 32, 16]
    ):

        super(DestripeNet, self).__init__()
        self.Destripe = nn.ModuleList()
        
        for feature in features:
            self.Destripe.append(
                nn.ConvTranspose1d(in_channels=in_channels, out_channels=feature, kernel_size=2, stride=2)
            )
            self.Destripe.append(
                nn.ReLU()
            )
            in_channels = feature
        
        self.Destripe_final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=131072, out_features=2532) # crops prediction (based on summed data length)
        )

    def forward(self, x):
        for layer in self.Destripe:
            x = layer(x)
        return self.Destripe_final(x)



        
            

