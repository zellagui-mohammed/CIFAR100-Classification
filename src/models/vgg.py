import torch.nn as nn 
import torch
class ConvBloc(nn.Module):

    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.bloc=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      stride=1,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self ,x ):

        return self.bloc(x)


class VGGBloc(nn.Module):

    def __init__(self,in_channels,out_channels,num_convs):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        bloc = []
        bloc.append(ConvBloc(in_channels,out_channels))
        for _ in range(1,num_convs):

            bloc.append(ConvBloc(out_channels,out_channels))
        self.layers = nn.Sequential(*bloc)
    def forward(self ,x ):

        x=self.layers(x)
        x=self.maxpool(x)
        return x 


class VGG(nn.Module):

    def __init__(self,in_channels,num_classes,config):
        super().__init__()
       

        stages = []
        out_channels=64
        current_in_channels=in_channels

        
        for i in range(0,len(config)):
            stages.append(VGGBloc(current_in_channels,out_channels,num_convs=config[i]))
            current_in_channels=out_channels
            out_channels=min(out_channels*2,512)
      
        self.backbone= nn.Sequential(*stages)

        self.adaptivepool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier=nn.Sequential(
            
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256,num_classes)
            )
        




    def forward(self,x):

        x=self.backbone(x)
        x=self.adaptivepool(x)
        x=torch.flatten(x,1)
        x=self.classifier(x)
        
        return x 
















