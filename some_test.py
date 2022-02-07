import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


N = 100
D = 64

X = torch.randn(N, D) # 100 example, of which is 64 dimension
y = torch.randn(N)

# Define the weight of the model

# Layer 1, 100 by 50
# Layer 2, 50 by 1
# For simplicity, I use a regression model
# I use torch.empty() since I will initialize them later. 
# We can also just use torch.randn() with a very small variance


# weight for layer 1
F1 = torch.empty(32, 64)
# bias for layer 1
BF1 = torch.empty(32)
 
# weight for layer 2
F2 = torch.empty(1, 32)
# bias for layer 2
BF2 = torch.empty(1)

# Init
nn.init.xavier_normal_(F1)
nn.init.constant_(BF1, 1 / BF1.numel())
nn.init.xavier_normal_(F2)
nn.init.constant_(BF2, 1 / BF2.numel())

F1.requires_grad = True
F2.requires_grad = True
BF1.requires_grad = True
BF2.requires_grad = True


max_iter = 10
lr = 0.1


# we can use optimizer as usual
weights = [F1, BF1, F2, BF2]
optimizer = optim.Adam(weights, lr=lr)

for i in range(max_iter):

	# forward pass, we can use functions in torch.nn.functional.
	# It has almost every operations we need


	# layer 1, we can also do tmp = X @ F1 + BF1
	tmp = F.linear(X, F1, BF1)
	tmp = F.relu(tmp)
	# layer 2
	out = F.linear(tmp, F2, BF2).flatten()

	# find loss

	loss = ((out - y) ** 2).mean()

	# same as usual
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	print(loss.item())


#########################################################
#########################################################
# Find the individual gradient

def forward(F1, BF1, F2, BF2):

	tmp = F.linear(X, F1, BF1)
	tmp = F.relu(tmp)
	out = F.linear(tmp, F2, BF2).flatten()

	# no average here since we need the individual gradient
	# loss is a vector of dimension N
	loss = (out - y) ** 2

	return loss

inputs = (F1, BF1, F2, BF2)


# please see here for more reference: 
# https://pytorch.org/docs/1.10.1/generated/torch.autograd.functional.jacobian.html
# the grammar is jacobian(forward_function, iuput)
# vectorize=True is very important, otherwise it will be the same speed as for loop

grads = torch.autograd.functional.jacobian(forward, inputs, vectorize=True)


# grads is a tuple containing four parts

F1_grad, BF1_grad, F2_grad, BF2_grad = grads

# this is 100 x 32 x 64, which contains the indivial grad wrt to 100 examples
# F1_grad[0] is the grad wrt to example 0
# F1_grad[1] is the grad wrt to example 1
# ...

print(F1_grad.shape)


###### VERY IMPORTANT ######

# Since F1, BF1, F2, BF2 are tensors require gradient, so when updating them we should turn off the gradient
# otherwise updating the gradient will be recognize as forward pass
# something like the following

with torch.no_grad():
	F1.add_(lr * F1_grad.mean(dim=0))

# we should also always use inplace operators
# instead of doing F1 = F1 + F2, we should use F1.add_(F2)
# in this case, it won't create issues, but if use optimizer in the very first part,
# things will be messed up.
# Since weights = [F1, F2], if we F1 = F1 + F2, weights[0] is not F1 anymore and optimizer won't work
# F1.add_(F2) is fine since weights[0] is still F1
# I think you might need do some clipping and do manual update so it's not very trick, but it is better to use inplace 
# operator for the weight












