import torch

x = torch.arange(12, dtype=torch.float32)
print(x)

print(f"x has {x.numel()} elements")

print(f"The shape of x is {x.shape}")

x = torch.arange(12, dtype=torch.float32)
print(f"X is now reshaped to:")
X = x.reshape(3, 4)
print(X)
print("Making   an array of  zeros:")
print(torch.zeros((2, 3, 4)))
print("or ones")
print(torch.ones((2, 3, 4)))

print("random content would be ")
print(torch.randn(3, 4))

print("Specifyiong the vals:")
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

print("[-1] selects the last row and [1:3] selects the second and third rows")
print(X[-1], X[1:3])

print("rewriting  elements in a matrix")
X[1, 2] = 17
print(X)


print("We can assign multiple els the same val")
X[:2, :] = 12
print(X)

print("We can exponate every element of  an array: ")
print(torch.exp(x))

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])

print("We can do operations on multiple vectors")
print(x / y)
print(x**y)
print(x == y)

print("We can join matrices")

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))
print("but their  dimesnions must be  equal!")

print("We can   sum   them!")
print(X.sum())

print("We can sum  elementwise binary ops  by  broadcasting")
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)
print(a + b)

before = id(Y)
print(id(Y) == before)
Y = Y + X
print(id(Y) == before)

before = id(X)
print(id(X) == before)
X += Y
print(id(X) == before)
