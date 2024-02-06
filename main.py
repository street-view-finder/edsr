from edsr import EDSR
import torch
from PIL import Image
import numpy as np
import cv2

model = EDSR(16, 64, 2, 255, 3, 1)
model.load_state_dict(torch.load('edsr_baseline_x2.pt', map_location=torch.device('cpu')))
edsr = model.eval()

print('init model done')

image = Image.open('laura.jpg')
lr = np.array(image.convert('RGB'))
lr = lr[::].astype(np.float32).transpose([2, 0, 1]) / 255.0
lr = torch.as_tensor(np.array([lr]))

print('process input done')

pred = edsr(lr)

print('make prediction done')

pred = pred.data.cpu().numpy()
pred = pred[0].transpose((1, 2, 0)) * 255.0
pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

print('process output done')

cv2.imwrite('result.jpg', pred)

