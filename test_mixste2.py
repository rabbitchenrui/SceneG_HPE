import torch
from common.mixste import MixSTE2

if __name__ == "__main__":
    input_2d = torch.zeros((4, 243, 17, 2))
    input_3d = torch.zeros((4, 243, 17, 3))
    t = 10
    img = torch.zeros((4, 48, 512))
    depth = torch.zeros((4, 48, 17))
    model = MixSTE2()
    output = model(input_2d, input_3d, t, img, depth)