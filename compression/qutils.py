import torch
import einops
import matplotlib.pyplot as plt
from compression.qmodules import Qconv

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


def quant_spec(x,b=-8.0,e=2.0):
    """
    # this is just a spec function..
    # register a method for quantization for each module. just to be safe
    """
    b = torch.as_tensor(b)
    e = torch.as_tensor(e)

    b = torch.relu(b)
    x_upscaled = x/torch.exp2(e)
    half = torch.exp2(b -1)
    x_clipped = torch.clip(x_upscaled,-1*half,half-1)
    x_round = torch.steRound(x_clipped)
    return torch.exp2(e) * x_round
   
