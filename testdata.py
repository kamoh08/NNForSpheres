import torch
import spheredata

N = 30000

test_data = -1000.0 + torch.rand(N, 3) * 2000.0

center1 = spheredata.center1
center2 = spheredata.center2

radius1 = spheredata.radius1
radius2 = spheredata.radius2

d1 = ((test_data - center1) ** 2).sum(dim=1)
d2 = ((test_data - center2) ** 2).sum(dim=1)

test_output = torch.full(
    (N,),
    3.0
)

test_output[d1 <= radius1 ** 2] = 1.0
test_output[d2 <= radius2 ** 2] = 2.0