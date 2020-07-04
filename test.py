import os

import torch
import mtrnn
import imageio
import numpy as np
import time 
test_path = "test_input"
test_save_path = "test_output"
save = True
test_imgs_path = os.listdir(test_path)
test_imgs_path.sort()
model = mtrnn.MTRNN().cuda()
model.load_state_dict(torch.load("model_best.pt"))

iter_num = 6

for test_img_path in test_imgs_path:
    img_dir = os.path.join(test_path,test_img_path)
    lr = imageio.imread(img_dir)
    lr = np.ascontiguousarray(lr.transpose((2, 0, 1)))
    lr = torch.from_numpy(lr).float().unsqueeze(0).cuda()
    feature_1 = lr.clone()
    feature_2 = lr.clone()    
    lr_d = lr.clone()
    output_module = []
    output_module.append(lr)
    st = time.time()
    for itera in range(iter_num):
        if itera != 0:
            lr_d = lr_d.data
            feature_1 = feature_1.data
            feature_2 = feature_2.data
        output = model([lr,lr_d,feature_1,feature_2])    
        lr_d,feature_1,feature_2= output                 
        output_module.append(lr_d.data)
    ed = time.time()
    print("Run time for single image: ",ed-st)
    if save:
        print("Saving----")
        for itera,output in enumerate(output_module):
            sr = output.data.clamp(0, 255).round()
            sr = np.transpose(sr.cpu().numpy()[0],(1,2,0))
            output_path = "%s/%02d/%s"%(test_save_path,itera,test_img_path)
            imageio.imwrite(output_path, sr.astype(np.uint8))
