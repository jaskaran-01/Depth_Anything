from depth_anything.dpt import DepthAnything
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np
import cv2
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import torch
from time import sleep
import time

encoder = 'vitl'  # can also be 'vitb' or 'vitl' 'vits'
model_id = 'depth_anything_{:}14'.format(encoder)
if torch.cuda.is_available():
    DEVICE='cuda'
    depth_anything = DepthAnything.from_pretrained(f'LiheYoung/{model_id}').to(DEVICE).eval()
else :
    DEVICE='cpu'
    depth_anything = DepthAnything.from_pretrained(f'LiheYoung/{model_id}')
print(DEVICE)
cap=cv2.VideoCapture(r'tcp://192.168.137.93:8081')
start_time = time.time()
frame_count = 0
while True:
    ret, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])



    h, w = image.shape[:-1]
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    depth = depth_anything(image)
        
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            
    depth = depth.cpu().detach().numpy().astype(np.uint8)
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    org = [0,0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    cv2.putText(depth_color,"hello", org, font, fontScale, color, thickness)
    cv2.putText(depth_color, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    print(depth)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        cap.release()
        cv2.destroyAllWindows()
        break
    cv2.imshow('Webcam',depth_color)

