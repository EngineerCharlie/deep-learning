import torch

x = torch.arange(4.0)
print(f"x={x}")

print("Before we can calc grad of y wrt x w need to store  it")
x.requires_grad_(True)
x.grad

print("First we define the function 2 * x ^T * x")
y = 2 * torch.dot(x, x)
print(y)

print("So to differentiate y  wrt x we do:")
y.backward()
print(x.grad)
print("manual diff of  the  function y we know is 4x")
print(x.grad == 4 * x)


print("To take  grads of anotyher function of x we first must 0 the grads")

x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

print("Lets do a diff function")

x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))
print(x.grad)


x.grad.zero_()
y = x * x
print(y)
u = y.detach()
###
# ££ This step creates a tensor u that holds the same data as y,
# but it is detached from the computation graph. That means any
# operation performed on u will not affect the gradients of any
# tensors that created it. Effectively, u is treated as a constant
# with respect to the autograd system.
###
print(u)
z = u * x
print(z)

z.sum().backward()
x.grad == u
