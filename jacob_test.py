import torch

def foo(x):
	return x ** 2

x = torch.randn(20000, 2)

x.requires_grad = True

print(x)

dx = torch.autograd.functional.jacobian(foo, x, vectorize=True)

print(dx.shape)