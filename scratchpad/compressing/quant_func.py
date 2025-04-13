import torch
import matplotlib.pyplot as plt

class STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        return torch.round(x)

    @staticmethod
    def backward(ctx,upstream):
        return upstream

steRound = STERound.apply

def qbits(x,b=torch.tensor(6),e=torch.tensor(-8)):
    b = torch.relu(b)
    x_scaled = x/torch.exp2(e) # scaling up (numbers most likely to be b/n -1 and 1)
    # clip the scaled number if number is rep between 2^8 to 2^8 then the bit depth is 8.
    x_clipped = torch.clip(x_scaled, -1 * torch.exp2(b-1), torch.exp2(b-1)-1)
    # rounding will stop from numbers to flying to greater precision
    x_round = steRound(x_clipped)
    result = torch.exp2(e) * x_round # scale back
    return result


def bad_round():
    #leaf
    x = torch.linspace(-0.08,0.08,10)
    print(f"{x=}")
    x.requires_grad_()

    b = torch.tensor(6.0)
    b.requires_grad_()


    e = torch.tensor(-8.0)
    e.requires_grad_()

    x_scaled = x/torch.exp2(e)
    x_scaled.retain_grad()
    print(f"{x_scaled=}")

    x_clipped = torch.clip(x_scaled,-1*torch.exp2(b-1), torch.exp2(b-1)-1)
    x_clipped.retain_grad()
    print(f"clipped range {-1*torch.exp2(b-1)}, {torch.exp2(b-1)-1}")
    print(f"{x_clipped=}")


    x_round = torch.round(x_clipped)
    x_round.retain_grad()
    print(f"{x_round=}")

    result = torch.exp2(e) * x_round
    result.retain_grad()
    print(f"{result=}")

    loss = result.sum()
    loss.backward()
    print("===========")
    print(f"{result.grad=}")
    print(f"{x_round.grad=}")
    print(f"{x_clipped.grad=}")
    print(f"{x_scaled.grad=}")
    print(f"{x.grad=}")
    print(f"{b.grad=}")
    print(f"{e.grad=}")



def inspect_gradient():
    x = torch.linspace(-1,1,10)
    print(f"{x=}")
    x.requires_grad_()

    b = torch.tensor(1.0)
    b.requires_grad_()
    brange = torch.exp2(b-1)
    brange.retain_grad()

    e = torch.tensor(-3.0)
    e.requires_grad_()

    x_scaled = x/torch.exp2(e)
    x_scaled.retain_grad()
    print(f"{x_scaled=}")

    x_clipped = torch.clip(x_scaled,-1*brange, brange-1)
    x_clipped.retain_grad()
    print(f"clipped range {-1*torch.exp2(b-1)}, {torch.exp2(b-1)-1}")
    print(f"{x_clipped=}")


    x_round = steRound(x_clipped)
    x_round.retain_grad()
    print(f"{x_round=}")

    result = torch.exp2(e) * x_round
    result.retain_grad()
    print(f"{result=}")

    loss = result.sum()
    loss.backward()

    # unit tests
    print("==========unit_tests===============")
    our_resultGrad = torch.ones_like(result)
    print(f"{result.grad=}\n {our_resultGrad=}")
    assert torch.allclose(result.grad, our_resultGrad), "hein?"
    print(f"Test passed.")

    our_roundGrad = torch.ones_like(result)*torch.exp2(e)
    print(f"{x_round.grad=}\n{our_roundGrad=}")
    assert torch.allclose(x_round.grad, our_roundGrad), "hein?"
    print("Test Passed")

    de_branch1 = result.grad * result * torch.log(torch.tensor(2.0))

    print(f"{x_clipped.grad=}\n{x_round.grad=}")
    assert torch.equal(x_clipped.grad,x_round.grad),"hein?"
    print("Test Passed")

    low_range = -1*torch.exp2(b-1)
    high_range = torch.exp2(b-1)-1
    dx_scaled_local = torch.where(((x_scaled >= low_range)&(x_scaled<=high_range)),1.0,0.0)
    our_xscaledGrad = x_clipped.grad * dx_scaled_local
    print(f"{x_scaled.grad=}\n{our_xscaledGrad=}")
    assert torch.allclose(x_scaled.grad, our_xscaledGrad), "hein?"
    print("Test Passed")

    our_xGrad = x_scaled.grad * (1/torch.exp2(e))
    print(f"{x.grad=}\n{our_xGrad=}")
    assert torch.allclose(x.grad,our_xGrad),"hein?"
    print("Test Passed")

    brangegrad_branch1= torch.sum(torch.where((x_scaled<=low_range),-1.0,0.0)) 
    brangegrad_branch2= torch.sum(torch.where((x_scaled>=high_range),1.0,0.0)) 
    local_brangeGrad = (brangegrad_branch1 + brangegrad_branch2)
    our_brangeGrad = torch.sum(x_scaled.grad * local_brangeGrad)
    print(f"{brange.grad=}{our_brangeGrad=}")
    assert torch.allclose(brange.grad,our_brangeGrad),"Hein?"
    print("Test Passed")

    our_bGrad = brange.grad* brange * torch.log(torch.tensor(2))
    print(f"{our_bGrad=} {b.grad=}")
    assert torch.allclose(b.grad, our_bGrad) , "Hein?"
    print("Test Passed")

    de_branch2 = x_scaled.grad *x_scaled* -1*torch.log(torch.tensor(2.0))
    print(f"{de_branch1=}\n{de_branch2=}")
    de = de_branch1 + de_branch2
    print(f"{e.grad=}\n{de.sum()=}")
    assert torch.allclose(e.grad,de.sum()),"Hein"
    print("Test Passed")
    print("All Test Passed!")


def inspect_clip():
    x = torch.arange(-30.0,31.0)
    x.requires_grad_()
    # j,k = -1*torch.tensor(2.0) , torch.tensor(8.0)
    j = torch.tensor(8.0)
    j.requires_grad_()
    # k.requires_grad_()
    y = torch.clip(x,-j,j)
    y.retain_grad()
    l = y.sum()
    l.backward()
    print(f"{y.grad=}")
    print(f"{j.grad=}")
    # print(f"{k.grad=}")
    print(f"{x.grad=}")

def inspect_max_grad():
    x = torch.arange(-10.0,10.0)
    x.requires_grad_()

    k = torch.tensor(4.0)
    k.requires_grad_()

    j = torch.tensor(-2.0)
    j.requires_grad_()

    y = torch.clamp(x,j,k)
    y.retain_grad()
    loss = y.sum()

    loss.backward()
    print(f"{x=},{y=}")
    print(f"{y.grad=}")
    print(f"{k.grad=}")
    print(f"{j.grad=}")
    print(f"{x.grad=}")







def main():
    # x = torch.linspace(-128,128,100)
    # b = torch.tensor(5.0)
    # e = torch.tensor(2.0)
    # out = qbits(x,b,e)
    # plt.plot(x,out)
    # plt.show()

    inspect_gradient()
    # bad_round()
    # inspect_clip()
    # inspect_max_grad()
    ...

if __name__ == "__main__":
    main()

