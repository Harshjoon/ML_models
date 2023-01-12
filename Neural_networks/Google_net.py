import torch
from torch import Tensor
from torch import nn

class GoogLeNet(nn.Module):

    def __init__(
            self,
            num_classes: int = 1000,
            aux_logits: bool = True,
            transform_input: bool = False,
            dropout: float = 0.2,
            dropout_aux: float = 0.7,
            var = "U"
    ) -> None:
        super(GoogLeNet, self).__init__()
        
        self.dropout_value = dropout
        #self.aux_logits = aux_logits
        #self.transform_input = transform_input

        self.fc1_ = nn.Linear(3,30)
        self.fc2_ = nn.Linear(30,300)
        self.fc3_ = nn.Linear(300,3520)
         
        self.conv1       = BasicConv2d(1, 8, kernel_size=(9, 5), stride=(2, 2), padding=(1, 1))
        self.maxpool1    = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)
        self.conv2       = BasicConv2d(8, 16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.maxpool2    = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)

        self.inception3a = Inception(16, 8, 4, 8, 4, 8, 8)
        self.maxpool3    = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)
        self.inception3b = Inception(32, 16, 8, 16, 8, 16, 16)
        self.maxpool4    = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)
        
        self.avgpool     = nn.AdaptiveAvgPool2d((1, 1))
        
        if self.dropout_value != None:
            self.dropout = nn.Dropout(self.dropout_value, True)
            
        self.fc1         = nn.Linear(7040, 2000)
        self.fc2         = nn.Linear(2000, 500)
        self.fc3         = nn.Linear(500, num_classes)
        
        self.tanh        = nn.Tanh()
        self.sigmoid     = nn.Sigmoid()
        

    def forward(self, x: Tensor, x_) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        out = self._forward_impl(x, x_)
        return out

    def _forward_impl(self, x: Tensor,x_):

        #x   = self._transform_input(x)

        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.maxpool2(out)
        
        out = self.inception3a(out)
        out = self.maxpool3(out)
        out = self.inception3b(out)
        out = self.maxpool4(out)
        
        out = torch.flatten(out, 1)
        
        if self.dropout_value != None:
            out = self.dropout(out)
        
        out_ = self.fc1_(x_)
        out_ = self.sigmoid(out_)
        out_ = self.fc2_(out_)
        out_ = self.sigmoid(out_)
        out_ = self.fc3_(out_)
        out_ = self.sigmoid(out_)

        out = torch.cat((out,out_[:,0]),dim=1)

        out = self.fc1(out)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = self.fc3(out)

        split_point = 5
        first_slice = out[:,0:split_point]
        second_slice = out[:,split_point:]
        tuple_of_activated_parts = (
            self.tanh(first_slice),
            self.sigmoid(second_slice)
        )
        
        out = torch.cat(tuple_of_activated_parts, dim=1)
        return out
    
    
    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.01, a=-2, b=2)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(True)

        
    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Inception(nn.Module):
    def __init__(
            self,
            in_channels: int,
            ch1x1: int,
            ch3x3red: int,
            ch3x3: int,
            ch5x5red: int,
            ch5x5: int,
            pool_proj: int,
    ):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

    def forward(self, x: Tensor) -> Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        out = [branch1, branch2, branch3, branch4]
        out = torch.cat(out, 1)
        return out

# use naive inception
# use maxpool(instead of average pool) after last inception and after flatten. 