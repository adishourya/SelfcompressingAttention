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
   
