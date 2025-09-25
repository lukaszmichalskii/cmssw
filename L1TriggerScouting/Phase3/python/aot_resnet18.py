import os
import torch
import torchvision.models as models

# Load pretrained model
model = models.resnet101(pretrained=True).eval()

# Put on correct device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Example input (with dynamic batch dim)
example_inputs = (torch.randn(8, 3, 224, 224, device=device),)

# Optional: specify dynamic batch dimension
batch_dim = torch.export.Dim("batch", min=1, max=128)

# Export with dynamic shapes if needed
with torch.no_grad():
    exported = torch.export.export(
        model,
        example_inputs,
        dynamic_shapes={"x": {0: batch_dim}}  # 'input' refers to input name
    )

    # AOT compile and generate a package
    output_path = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=os.path.join(os.getcwd(), f"resnet18_{device}.pt2"),
        inductor_configs={"aot_inductor.package_cpp_only": True}
    )

print("AOT package path:", output_path)
