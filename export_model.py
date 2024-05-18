import torch
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("midas.torchscript.pt")
