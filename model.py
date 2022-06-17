import torch
from monai.networks.nets import resnet18 

class ResMgmtNET(torch.nn.Module):
    def __init__(self, pretrained = False):
        super().__init__()
        self.model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=1)
        if pretrained:
            net_dict = self.model.state_dict()
            pretrain = torch.load("../input/medical-net-files/pretrain/resnet_18_23dataset.pth")
            pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
            missing = tuple({k for k in net_dict.keys() if k not in pretrain['state_dict']})
            inside = tuple({k for k in pretrain['state_dict'] if k in net_dict.keys()})
            unused = tuple({k for k in pretrain['state_dict'] if k not in net_dict.keys()})

            pretrain['state_dict'] = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
            self.model.load_state_dict(pretrain['state_dict'], strict=False)
            
        self.depth = torch.nn.Linear(400, 128)
        self.last = torch.nn.Linear(128, 1)
        
    def forward(self, image):               # image, features can be added later 
        x = self.model(image)
        x = self.depth(x)
        x = self.last(x)
        return x
