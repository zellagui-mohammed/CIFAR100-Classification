import torch.nn as nn 


class LeNet(nn.Module):
    def __init__(self, nb_out,*args, **kwargs):
        super().__init__(*args, **kwargs)
        # la premier couche de convolution  avec le max pooling 
        self.conv1=nn.Conv2d(3,6,kernel_size=5)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)
        # la deuxiéme couche de convolution avec la max pooling 
        self.conv2=nn.Conv2d(6,16,kernel_size=5)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)
        # une convolution prèsque miel que le Flaten -> FC 
        self.conv3=nn.Conv2d(16,120,kernel_size=5)
        self.flatten=nn.Flatten()
        self.fc2=nn.Linear(120,84)
        # Fonction d'activation 
        self.fc3=nn.Linear(84,nb_out)

        self.tanh=nn.Tanh()
    def forward(self,x):
        x=self.conv1(x)
        x=self.tanh(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.tanh(x)
        x=self.pool2(x)
        x=self.conv3(x)
        x=self.tanh(x)
        x=self.flatten(x)
        x=self.fc2(x)
        x=self.tanh(x)
        x=self.fc3(x)

        return x 

        
