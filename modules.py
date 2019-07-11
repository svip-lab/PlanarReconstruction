import torch
import numpy as np

# define K for PlaneNet dataset
focal_length = 517.97
offset_x = 320
offset_y = 240

K = [[focal_length, 0, offset_x],
     [0, focal_length, offset_y],
     [0, 0, 1]]
K_inv = np.linalg.inv(np.array(K))

if torch.cuda.is_available():
    K = torch.FloatTensor(K).cuda()
    K_inv = torch.FloatTensor(K_inv).cuda()
else:
    K = torch.FloatTensor(K)
    K_inv = torch.FloatTensor(K_inv)

h, w = 192, 256

x = torch.arange(w, dtype=torch.float32).view(1, w) / w * 640
y = torch.arange(h, dtype=torch.float32).view(h, 1) / h * 480

if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()

xx = x.repeat(h, 1)
yy = y.repeat(1, w)

if torch.cuda.is_available():
    xy1 = torch.stack((xx, yy, torch.ones((h, w), dtype=torch.float32).cuda()))   # (3, h, w)
else:
    xy1 = torch.stack((xx, yy, torch.ones((h, w), dtype=torch.float32)))   # (3, h, w)

xy1 = xy1.view(3, -1)                                    # (3, h*w)

k_inv_dot_xy1 = torch.matmul(K_inv, xy1)  # (3, h*w)
