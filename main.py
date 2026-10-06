import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import traindata
import testdata

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

class MyNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(6, 2)
        self.linear2 = nn.Linear(2, 2)
        self.linear3 = nn.Linear(2, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x / 1000.0

        features = torch.cat(
            (x, x ** 2),
            dim=1
        )

        distances = self.linear1(features)

        difference = self.linear2(distances)

        inside = torch.sigmoid(167 * difference)

        final_output = self.linear3(inside)

        return distances, difference, final_output


model = MyNN().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)


train_data = traindata.train_input.to(device)
true_distance = (traindata.Truedistance.to(device) / 1000.0 ** 2)
true_difference = (traindata.Truedifference.to(device) / 1000.0 ** 2)
train_output = (traindata.train_output.to(device))

test_data = (testdata.test_data.to(device))
test_output = (testdata.test_output.to(device))


batch_size = 12500
epochs = 3000


for epoch in range(epochs):

    # model.train()

    epoch_loss = 0.0
    batches = 0

    for start in range(0,len(train_data),batch_size):

        end = start + batch_size

        x_batch = train_data[start:end]
        dist_batch = true_distance[start:end]
        diff_batch = true_difference[start:end]

        output_batch = (train_output[start:end].unsqueeze(1))

        optimizer.zero_grad(set_to_none=True)

        distances, difference, output = model(x_batch)


        loss1 = loss_fn(distances,dist_batch)

        loss2 = loss_fn(difference,diff_batch)

        loss3 = loss_fn(output, output_batch)

        loss = loss1 + loss2 + loss3

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        batches += 1


    if epoch % 100 == 0:
        print(
            f"epoch {epoch} | "
            f"loss1={loss1.item():.6f} | "
            f"loss2={loss2.item():.6f} | "
            f"loss3={loss3.item():.6f} | "
            f"average={epoch_loss / batches:.6f}"
        )


    if epoch % 500 == 0:

        # model.eval()

        with torch.inference_mode():

            _, _, visual_output = model(
                test_data
            )

            values = visual_output.squeeze(1)

            sphere1_mask = (values < 1.5)

            sphere2_mask = ((values >= 1.5) & (values < 2.5))

            sphere1_points = (test_data[sphere1_mask].cpu().numpy())

            sphere2_points = (test_data[sphere2_mask].cpu().numpy())

        fig = plt.figure()

        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            sphere1_points[:, 0],
            sphere1_points[:, 1],
            sphere1_points[:, 2],
            s=3,
            label="Sphere 1"
        )

        ax.scatter(
            sphere2_points[:, 0],
            sphere2_points[:, 1],
            sphere2_points[:, 2],
            s=3,
            label="Sphere 2"
        )

        ax.set_xlim(-1000,1000)
        ax.set_ylim(-1000,1000)
        ax.set_zlim(-1000,1000)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_title(f"Epoch {epoch}")

        ax.legend()
        plt.show()



with torch.inference_mode():

    _, _, output = model(
        test_data
    )

    values = output.squeeze(1)


    predictions = torch.full_like(
        values,
        3.0
    )


    predictions[values < 2.5] = 2.0

    predictions[values < 1.5] = 1.0

    accuracy = (predictions == test_output).float().mean() * 100

print(
    f"overall accuracy = "
    f"{accuracy.item():.2f}%"
)