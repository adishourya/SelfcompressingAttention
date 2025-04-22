import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.gelu = torch.nn.functional.gelu
torch.mul_reduce = torch.prod


class STERound(torch.autograd.Function):
    """
    same gradient as parent node
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


@torch.no_grad
def inspect_qconv_weights(out,layer):
    assert isinstance(layer,Qconv), "Hein? should be Qconv layer"
    out = out.to("cpu")
    kernel = layer._quantized_weight().to("cpu")
    _,in_channels,k,_ = kernel.shape
    if in_channels>1:
        kernel = kernel[:,0,:,:][:,None,:,:]
    
    kernel_plot = einops.rearrange(kernel,"out_ch in_ch k1 k2 ->  (in_ch k1) (out_ch k2)")
    out_plot = einops.rearrange(out,"b c h w -> (b h) (c w)")


    plt.figure(figsize=(15,8))
    plt.imshow(kernel_plot,cmap="gray")
    plt.show()
    plt.figure(figsize=(15,8))
    plt.imshow(out_plot,cmap="Blues")
    plt.show()
    return out_plot,kernel_plot
   

class QTrainer:
    def __init__(self,model):
        self.model = torch.compile(model.to("cuda"))
        self.track_decay = []
        self.track_activekernels = []
        self.track_loss = []

    
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            weight_decay=1e-3)
    
        self.gamma = (1/10) # high for drama!! but should be around 0.05 or something.. compression factor
        # we need to calculate total number of parameters at initialization (papaer calls it N)
        # here since everything is trainable

        self.tot_init = sum(p.numel() for group in self.optim.param_groups for p in group['params'] if p.requires_grad)
        self.tot_qparams = torch.sum(torch.tensor([p_weight.numel() for p,p_weight in self.model.named_parameters() if "_bit" in p]))
    
        print(f"Total Parameters {self.tot_init=}")
        print(f"of which compression are :{self.tot_qparams=}")
        print(f"compression factor at init {self.gamma * self._qlayersize()}")
    
        print(self._qlayersize())
        print(self._activekernelscount())


    def _qlayersize(self):
        return torch.sum(torch.tensor([layer.size_layer() for layer in self.model.modules() if isinstance(layer,Qconv)]))/self.tot_init


    def _activekernelscount(self):
        kernel_counts = dict()
        for name,layer in self.model.named_modules():
            if isinstance(layer,Qconv):
                depths = torch.relu(layer.depth_bit)
                count =torch.sum(torch.where(depths>0,1,0)).item()
                kernel_counts[name] = count
        return kernel_counts
        

    
    # @torch.compile
    def train(self,num_epochs=10):
        pbar_epoch = tqdm(range(num_epochs))
        for epoch in pbar_epoch:
            i = 0
            for batch_img, batch_label in train_loader:
                batch_img = batch_img.to("cuda")
                batch_label = batch_label.to("cuda")
                out = self.model(batch_img)
                bit_decay = self._qlayersize()
                loss = torch.nn.functional.cross_entropy(input=out,target=batch_label) + self.gamma * bit_decay
                self.optim.zero_grad() # commented (bugged) out to show quick grad accum which drops conv filters
                loss.backward()
                self.optim.step()
                i = i +1
                if i %50 == 0:
                    activekernels = self._activekernelscount()
                    pbar_epoch.set_postfix(
                        loss=loss.item(),
                        decay=self.gamma*bit_decay.item(),
                        activekernels = activekernels,
                    )
                    self.track_activekernels.append(activekernels)
                    self.track_decay.append(bit_decay.item())
                    self.track_loss.append(loss.item())

        return self.model
    
