from torch import nn
import torch
from model_helper import get_conv_layer, get_activation, DownBlock

class LocNet(nn.Module):
    """
    activation: 'relu', 'leaky', 'elu'
    normalization: 'batch', 'instance', 'group{group_size}'
    conv_mode: 'same', 'valid'
    dim: 2, 3
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 2,
                 input_image_size: int = 64,
                 n_blocks: int = 4,
                 start_filters: int = 32,
                 activation: str = 'relu',
                 normalization: str = 'batch',
                 conv_mode: str = 'same',
                 dim: int = 2,
                 up_mode: str = 'transposed', 
                 #final_activation = None
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.start_filters = start_filters
        self.activation = activation
        self.normalization = normalization
        self.conv_mode = conv_mode
        self.dim = dim
        self.up_mode = up_mode

        self.down_blocks = []
        self.up_blocks = []

        size = int(input_image_size / (2 ** n_blocks ))
        n_filters = int(start_filters * (2 ** (n_blocks-1)))
        n_nodes = int (n_filters * (size ** dim))
        
        # print('size=', size)
        # print('n_filters=', n_filters)
        # print('n_nodes=', n_nodes)

        self.n_nodes_for_first_linear = n_nodes

        self.fc1 = nn.Linear(n_nodes, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, dim * 2)
        
        self.ac1 = get_activation(activation)
        self.ac2 = get_activation(activation)
        self.ac3 = nn.Sigmoid()
        #self.final_activation = final_activation

        # create encoder path
        for i in range(self.n_blocks):
            num_filters_in = self.in_channels if i == 0 else num_filters_out
            num_filters_out = self.start_filters * (2 ** i)
            #pooling = True if i < self.n_blocks - 1 else False

            down_block = DownBlock(in_channels=num_filters_in,
                                   out_channels=num_filters_out,
                                   pooling=True,
                                   activation=self.activation,
                                   normalization=self.normalization,
                                   conv_mode=self.conv_mode,
                                   dim=self.dim,
                                   two_conv_layers=False)

            self.down_blocks.append(down_block)

        # add the list of modules to current module
        self.down_blocks = nn.ModuleList(self.down_blocks)
        
        # initialize the weights
        self.initialize_parameters()

    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.weight, **kwargs)  # weights

    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.bias, **kwargs)  # bias

    def initialize_parameters(self,
                              method_weights=nn.init.xavier_uniform_,
                              method_bias=nn.init.zeros_,
                              kwargs_weights={},
                              kwargs_bias={}
                              ):
        for module in self.modules():
            self.weight_init(module, method_weights, **kwargs_weights)  # initialize weights
            self.bias_init(module, method_bias, **kwargs_bias)  # initialize bias

    def forward(self, x: torch.tensor):
        # Encoder pathway
        for module in self.down_blocks:
            x, before_pooling = module(x)

        #print('x.shape before flatten==>', x.shape)

        # flatten        
        x = x.view(-1, self.n_nodes_for_first_linear)

        #print('x.shape after CNN==>', x.shape)

        x = self.ac1(self.fc1(x))

        #print('x.shape after ac1==>', x.shape)

        x = self.ac2(self.fc2(x))

        #print('x.shape after ac2==>', x.shape)

        x = self.ac3(self.fc3(x))

        #print('x.shape after ac3==>', x.shape)

        return x

    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys() if '_' not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'
