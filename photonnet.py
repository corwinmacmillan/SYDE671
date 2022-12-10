import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):   
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__() 
        self.conv = nn.Sequential(
            # First conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                # Note: bias=False as we are using nn.BatchNorm2d() -> bias get cancelled by batch norm
            nn.BatchNorm2d(out_channels),
                # nn.BatchNorm2d(): Applies a batch normailzation by re-centering and re-scaling layers - see https://en.wikipedia.org/wiki/Batch_normalization
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            
            # Second conv layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class PhotonNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        # in_channels: Number of channels input to UNet
        # out_channels: Number of channels output of UNet -> we choose 1 as we are doing binary classification
        # features: The features of the down sampling and up sampling (ie. number of channels in a layer of a level)

        super(PhotonNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature)) # Acting as a list
            in_channels = feature
        
        # Up part of UNet (Note: We using transpose convolutions (has some artifacts) -> could also use a bilinear transformation and conv layer after)
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, out_channels=feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, out_channels=feature))

        # Bottom of UNet
        self.bottleneck = DoubleConv(features[-1], features[-1]*2) #512 channels 

        # Final UNet conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    

    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        
        # Reverse the skip connections list as we are concatenating the last one to the first upsample
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x) # Even i is nn.ConvTranspose2d()
            skip_connection = skip_connections[i//2] 

            # Resize x in case max pool floor sizing (input 161x161 -> max pool -> output 80x80)
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1) # Concatenate the data
            x = self.ups[i+1](concat_skip) # Odd i is DoubleConv()

        return self.final_conv(x)