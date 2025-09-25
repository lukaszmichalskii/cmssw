# @see: https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html

import os
import torch
import torch.nn as nn


class BenchmarkMLP(nn.Module):
    def __init__(self, input_dim=16, hidden_dims=[32, 64], output_dim=10):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)
    

with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BenchmarkMLP().to(device=device)
    example_inputs=(torch.randn(8, 16, device=device),)
    batch_dim = torch.export.Dim("batch", min=1, max=1024)
    # [Optional] Specify the first dimension of the input x as dynamic.
    exported = torch.export.export(model, example_inputs, dynamic_shapes={"input": {0: batch_dim}})
    # [Note] In this example we directly feed the exported module to aoti_compile_and_package.
    # Depending on your use case, e.g. if your training platform and inference platform
    # are different, you may choose to save the exported model using torch.export.save and
    # then load it back using torch.export.load on your inference platform to run AOT compilation.
    output_path = torch._inductor.aoti_compile_and_package(
        exported,
        # [Optional] Specify the generated shared library path. If not specified,
        # the generated artifact is stored in your system temp directory.
        package_path=os.path.join(os.getcwd(), f"model_small_{device}.pt2"),
        inductor_configs={"aot_inductor.package_cpp_only": True}
    )
