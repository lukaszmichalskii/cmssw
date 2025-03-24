import sys
import os
import torch

# prepare the datadir
if len(sys.argv) >= 2:
    datadir = sys.argv[1]
else:
    thisdir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(os.path.dirname(thisdir), "bin", "data")

os.makedirs(datadir, exist_ok=True)

class MultiplyByTwoModel(torch.nn.Module):
    def forward(self, input):
        return input * 2

# Instantiate the model
module = MultiplyByTwoModel()
x = torch.tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
print(module(x))

tm = torch.jit.trace(module.eval(), x)
tm.save(f"{datadir}/model_x2.pt")

print("model_x2.pt created successfully!")
