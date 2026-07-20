import utils.centernet as cennet
import cv2
import json
import numpy as np
import torch
import model_zoo.LaneNet as LaneNet
from Configs import config as cfg
from evaluation.evaluation_Tusimple import LaneEval
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

def forevaluation(config):
    tusimple_root = Path(config.TusimpleTesting_root)
    inputW = cfg.Dataprocess_cfg.imgSize[1]
    inputH = cfg.Dataprocess_cfg.imgSize[0]
    model = LaneNet.LaneNet(config.The_selected_backbone, config)
    tempmodel = torch.load(config.testmodel, map_location="cuda:0")
    model.load_state_dict(tempmodel)
    print("loaded")
    model.cuda()
    model.eval()
    with torch.no_grad():
        raw_data=[]
        with (tusimple_root / 'test_tasks_0627.json').open("r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                temp = json.loads(line)
                raw_data.append(temp)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.361, 0.405, 0.406], std=[0.192, 0.214, 0.232])
        test_json=raw_data
        colors = [
            [0, 0, 255],
            [0, 255, 0],
            [0, 165, 255],
            [0, 255, 255],
            [255, 0, 255],
            [255, 255, 0],
            [128, 0, 128],
            [0, 165, 255],
            [128, 128, 128],
            [0, 0, 128]
        ]
        cv2.namedWindow("final", cv2.WINDOW_NORMAL)
        for i in range(0, len(raw_data)):
            print(">>>processing the %d(%d)th image" % (i, len(raw_data)))
            image0 = Image.open(tusimple_root / raw_data[i]["raw_file"]).convert('RGB')
            image_width, image_height = image0.size
            input = image0.resize((inputW, inputH))
            input = to_tensor(input)
            input = normalize(input)
            input = input.unsqueeze(0).cuda()
            output = model.forward(input)
            segres = output[1]
            seg_res = segres[0, :, :, :]
            seg_res = seg_res.cpu().squeeze(0)
            seg_res = seg_res.detach().numpy().transpose((1, 2, 0))
            out = seg_res
            out = np.argmax(out, axis=2)
            insres = output[0]
            insres = insres / insres.norm(dim=1).unsqueeze(1)
            ins_res = insres[0, :, :, :]
            ins_res = ins_res.cpu().squeeze(0)
            ins_res = ins_res.detach().numpy().transpose((1, 2, 0))
            cen_seg_res = output[2]
            cen_seg_res = cennet._nms(cen_seg_res, 11)
            _, _, _, _, _, total_center = cennet._topk(cen_seg_res, ins_res, K=20)
            weights = torch.empty(len(total_center), ins_res.shape[2], 1, 1).cuda()
            for m in range(0, len(total_center)):
                weights[m, :, 0, 0] = torch.from_numpy(total_center[m].transpose()).cuda()
            classificationres = torch.nn.functional.conv2d(insres, weights)[0].cpu().detach().numpy()
            tempc = np.argmax(classificationres, axis=0)
            tempv = np.max(classificationres, axis=0)
            image = tensor_to_cv2image(image0)
            for m in range(0, weights.shape[0]):
                color = colors[m % len(colors)]
                mask1 = tempc == m
                mask2 = out == 1
                mask3 = tempv > config.CosThresh2
                mask = mask1 & mask2 & mask3
                mask = cv2.resize(mask.astype(np.uint8), dsize=(image_width, image_height),
                                   interpolation=cv2.INTER_NEAREST)
                pos = np.stack(np.where(mask == 1))
                if pos.shape[1] < config.minNum:
                    continue
                idx = np.argsort(pos[0, :], axis=0)
                pos[0, :] = pos[0, idx]
                pos[1, :] = pos[1, idx]
                x = raw_data[i]['h_samples']
                showy = pos[0, :]
                minidx = np.argmin(showy)
                showx = pos[1, :]
                minpos = showy[minidx]
                prex = []
                for n in range(0, len(x)):
                    idx = np.where(showy == x[n])
                    if x[n] <= minpos:
                        prex.append(-2)
                    elif np.size(idx):
                        posx = np.mean(showx[idx[0]])
                        posy = x[n]
                        cv2.line(image, (int(posx), posy), (int(posx), posy), color, thickness=10)
                        if posx < 0 or posx > image_width:
                            prex.append(-2)
                        else:
                            prex.append(posx.astype(float))
                    else:
                        higheridx = np.where(showy < x[n])
                        loweridx = np.where(showy > x[n])
                        if np.size(higheridx) and np.size(loweridx):
                            hidx = len(higheridx) - 1
                            lidx = len(loweridx) - 1
                            hx = showx[higheridx[0][hidx]]
                            hy = showy[higheridx[0][hidx]]
                            lx = showx[loweridx[0][lidx]]
                            ly = showy[loweridx[0][lidx]]
                            k = (hx - lx) * 1.0 / (hy - ly)
                            posx = int(k * (x[n] - hy) + hx)
                            posy = x[n]
                            cv2.line(image, (int(posx), posy), (int(posx), posy), color, thickness=10)
                            if posx < 0 or posx > image_width:
                                prex.append(-2)
                            else:
                                prex.append(posx)
                        elif np.size(higheridx):
                            f2 = np.polyfit(pos[0, pos.shape[1] // 2:pos.shape[1]],
                                            pos[1, pos.shape[1] // 2:pos.shape[1]], 1)
                            posx = np.polyval(f2, x[n])
                            posy = x[n]
                            cv2.line(image, (int(posx), posy), (int(posx), posy), color, thickness=10)
                            if posx < 0 or posx > image_width:
                                prex.append(-2)
                            else:
                                prex.append(posx)
                        elif np.size(loweridx) and x[n] >= 260:
                            f2 = np.polyfit(pos[0, 0:pos.shape[1] // 5], pos[1, 0:pos.shape[1] // 5], 1)
                            posx = np.polyval(f2, x[n])
                            posy = x[n]
                            cv2.line(image, (int(posx), posy), (int(posx), posy), color, thickness=10)
                            if posx < 0 or posx > image_width:
                                prex.append(-2)
                            else:
                                prex.append(posx)
                        else:
                            prex.append(-2)
                test_json[i]["lanes"].append(prex)
            test_json[i]['run_time'] = 1
            cv2.imshow("final", image)
            cv2.waitKey(1)
        save_result(test_json, "test_result.json")
    print(LaneEval.bench_one_submit("test_result.json", "test_label.json"))

def tensor_to_cv2image(img):
    image_np = np.array(img)
    image_np = image_np[..., ::-1]
    return image_np.copy()

def save_result(result_data, fname):
    with open(fname, 'w') as make_file:
        for i in result_data:
            json.dump(i, make_file, separators=(',', ': '))
            make_file.write("\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--CosThresh2", default=0.8, type=float)
    parser.add_argument("--minNum", default=1000, type=int)
    parser.add_argument("--The_selected_backbone", default='SlimResNet',
                        choices=['SlimResNet'])
    parser.add_argument("--TusimpleTesting_root", default="/path/to/tusimple/test_set/", type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--testmodel', default="./model.pkl")
    config = parser.parse_args()
    torch.cuda.set_device(config.device)
    forevaluation(config)
