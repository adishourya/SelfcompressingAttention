import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.gelu = torch.nn.functional.gelu
torch.mul_reduce = torch.prod


class STERound(torch.autograd.Function):
    """
    same gradient as parent node. Also called as pass through gradient.
    """
    @staticmethod
    def forward(ctx,x):
        return torch.round(x)

    @staticmethod
    def backward(ctx,upstream):
        """
        local jacobian is 1.
        """
        return upstream

torch.steRound = STERound.apply


class Qconv(torch.nn.Module):
    def  __init__ (self,in_channels,out_channels,kernel_size=3,depth=-8,scale=1.5):
        """
        # weight tying exp_bits and depth_bits
        # note number  of output  channels  is number of filterkernels  launched
        # we will try to not just compress  but take out entire filter kernels...
        # these asserts them to be pytorch tensors
        """
        super().__init__()
       
        in_channels = torch.as_tensor(in_channels)
        out_channels = torch.as_tensor(out_channels)
        kernel_size = torch.as_tensor(kernel_size)
        depth = torch.as_tensor(depth)
        scale= torch.as_tensor(scale)

        # fan_in is just in_channels
        weight_scale = 1/ torch.sqrt(in_channels*out_channels*out_channels)
        self.weight = torch.ones(out_channels,in_channels,kernel_size,kernel_size)
        self.weight = self.weight.uniform_(-weight_scale,weight_scale)

        # 1 for each kernel (out_channel).. to perform safe broadcasting we fill the rest of them with 1
        self.exp_bit = torch.ones(out_channels,1,1,1)*-8.0
        self.depth_bit = torch.ones(out_channels,1,1,1)*2

        # exp and depth also as trainables
        self.weight = torch.nn.Parameter(self.weight)
        self.exp_bit = torch.nn.Parameter(self.exp_bit)
        self.depth_bit = torch.nn.Parameter(self.depth_bit)
        ...
    
    def size_layer(self):
        """
        given by equation 4 : I*H*W * sum(b(i,l)
        Where O , I , H and W are the output, input, height, and
        width dimensions (so shape) of the weight tensor of layer l respec-
        tively, and b(i,l) is the bit depth of output channel i of layer l.
        """
        prods = torch.as_tensor(self.weight.shape[1:])
        size = torch.mul_reduce(prods) *  torch.sum(torch.relu(self.depth_bit))
        return size

    def _quantized_weight(self):
        b = torch.relu(self.depth_bit)
        x_upscaled = self.weight/torch.exp2(self.exp_bit)
        half = torch.exp2(b -1)
        x_clipped = torch.clip(x_upscaled,-1*half,half-1)
        x_round = torch.steRound(x_clipped)
        return torch.exp2(self.exp_bit) * x_round

    

    def __call__(self,x):
        # quantize every forward pass
        W = self._quantized_weight()
        # assert self.weight.shape==W.shape
        # valid padding or should we do same.. paper does not say
        return torch.nn.functional.conv2d(x,W,padding=1)

