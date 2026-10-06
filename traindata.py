import torch
import spheredata

N = 150000

train_input = -1000.0 + torch.rand(N, 3) * 2000.0

center1 = spheredata.center1
center2 = spheredata.center2

radius1 = spheredata.radius1
radius2 = spheredata.radius2

d1 = ((train_input - center1) ** 2).sum(dim=1)
d2 = ((train_input - center2) ** 2).sum(dim=1)

Truedistance = torch.stack(
    (d1, d2),
    dim=1
)

Truedifference = torch.stack(
    (
        radius1 ** 2 - d1,
        radius2 ** 2 - d2
    ),
    dim=1
)

train_output = torch.full(
    (N,),
    3.0
)

train_output[d1 <= radius1 ** 2] = 1.0
train_output[d2 <= radius2 ** 2] = 2.0