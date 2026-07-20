from torch.utils.data import Dataset
import numpy as np
import cv2
from Configs import config as cfg
import json
import random
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from Data.transforms import Transforms

TUSIMPLE_SPLIT_SEED = 123

def load_tusimple_annotations(root):
    root = Path(root)
    label_files = sorted(root.glob("label_data_*.json"))
    if not label_files:
        raise FileNotFoundError(f"No TuSimple label_data_*.json files found in {root}")
    data = []
    for label_file in label_files:
        with label_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    rng = random.Random(TUSIMPLE_SPLIT_SEED)
    rng.shuffle(data)
    return data

class LaneDataset(Dataset):
    def __init__(self, path):
        self.root = Path(path)
        self.transform = Transforms(inputW=cfg.Dataprocess_cfg.imgSize[1], inputH=cfg.Dataprocess_cfg.imgSize[0])
        self.transform.settings()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.361,0.405,0.406], std=[0.192,0.214,0.232])
        data = load_tusimple_annotations(self.root)
        self.raw_data = []
        for j in range(0, len(data)):
            if j % 50 == 0:
                continue
            self.raw_data.append(data[j])
        print("We have %d annotated image" % (len(self.raw_data)))

    def get_cen(self, anno):
        center_rows = []
        center_cols = []
        for i in reversed(range(len(anno['lanes']))):
            cols = (np.array(anno['lanes'][i]).astype(float))
            rows = (np.array(anno['h_samples']).astype(float))
            mask = cols >= 0
            cols = cols[mask]
            rows = rows[mask]
            if rows.shape[0] == 0:
                del anno['lanes'][i]
                continue
            if rows[0] > rows[-1]:
                cols = cols[::-1]
            unique_idx = np.sort(np.unique(rows, return_index=True)[1])
            cols = cols[unique_idx]
            rows = rows[unique_idx]
            p1 = np.polyfit(rows, cols, 3)
            mask1 = 1280 > cols
            mask2 = cols >= 0
            mask3 = 720 > rows
            mask4 = rows >= 0
            cols[~mask1 | ~mask2| ~mask3| ~mask4] = -2
            filtered_cols = cols[cols != -2]
            filtered_rows = rows[cols != -2]
            if len(filtered_cols) == 0:
                print("No valid lane points remain after filtering while computing lane centers.")
                continue
            maxidx = np.argmax(filtered_rows)
            minidx = np.argmin(filtered_rows)
            center_row = 0.5 * (filtered_rows[minidx] + filtered_rows[maxidx])
            center_col = np.polyval(p1, center_row)
            center_rows.insert(0, center_row)
            center_cols.insert(0, center_col.item())
        center_point = list(zip(center_cols, center_rows))
        return center_point

    def get_data_org(self, idx):
        img = Image.open(self.root / self.raw_data[idx]["raw_file"]).convert('RGB')
        anno = self.raw_data[idx]
        cen = self.get_cen(anno)
        return img, anno, cen

    def get_data_aug(self, img, anno, cen):
        img_new, lane_new, cen_new = self.transform.process(img, anno, cen)
        img_new = Image.fromarray(img_new)
        img_new = self.to_tensor(img_new)
        return {'img': self.normalize(img_new),
                'lanes': lane_new,
                'cens': cen_new}

    def get_label_map(self, data):
        out = dict()
        self.label = dict()
        self.label['cen'] = np.zeros((cfg.Dataprocess_cfg.imgSize[0], cfg.Dataprocess_cfg.imgSize[1]))
        self.label['ins'] = np.zeros((cfg.Dataprocess_cfg.imgSize[0], cfg.Dataprocess_cfg.imgSize[1]))
        self.label['seg'] = np.zeros((cfg.Dataprocess_cfg.imgSize[0], cfg.Dataprocess_cfg.imgSize[1]))
        if not (len(data['lanes']) == len(data['cens'])):
            print("Inconsistent lane and center counts; returning empty labels.")
            out.update(self.label)
            return out
        laneidx=0
        for i in range(len(data['lanes'])):
            laneidx+=1
            lane_pts = data['lanes'][i].astype('int16')
            cen_pts = data['cens'][i]
            self.label['cen'] = cv2.circle(self.label['cen'], (int(cen_pts[0]), int(cen_pts[1])), color=[1, 1, 1], radius=1, thickness=-1)
            for num in range(0, len(lane_pts) - 1):
                if lane_pts[num][0] > 0 and lane_pts[num + 1][0] > 0:
                    cv2.line(self.label['seg'], (lane_pts[num][0], lane_pts[num][1]), (lane_pts[num + 1][0], lane_pts[num + 1][1]),
                             [1, 1, 1], cfg.Dataprocess_cfg.lane_width['seg'])
                    cv2.line(self.label['ins'], (lane_pts[num][0], lane_pts[num][1]), (lane_pts[num + 1][0], lane_pts[num + 1][1]),
                             [laneidx, laneidx, laneidx], cfg.Dataprocess_cfg.lane_width['ins'], cv2.LINE_AA)
        self.label['seg'] = np.int8(self.label['seg'] != 0)
        self.label['ins'] = np.int8(self.label['ins'])
        point = np.where(self.label['cen'] == 1)
        point = np.stack(point)
        for idx in range(0, point.shape[1]):
            self.label['cen'] = draw_gaussian(self.label['cen'], (point[1, idx], point[0, idx]), 10)
        out.update(self.label)
        return out

    def remove_dict_keys(self, data):
        keylist = ['lanes','cens']
        for key in keylist:
            data.pop(key)
        return data

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        out = dict()
        img, anno, cen = self.get_data_org(idx)
        out.update(self.get_data_aug(img, anno, cen))
        out.update(self.get_label_map(out))
        out = self.remove_dict_keys(out)
        return out['img'],out['ins'],out['seg'],out['cen']

class LaneDataset_Test(Dataset):
    def __init__(self, path):
        self.root = Path(path)
        self.transform = Transforms(inputW=cfg.Dataprocess_cfg.imgSize[1], inputH=cfg.Dataprocess_cfg.imgSize[0])
        self.transform.settings()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.361,0.405,0.406], std=[0.192,0.214,0.232])
        data = load_tusimple_annotations(self.root)
        self.raw_data = []
        for j in range(0, len(data)):
            if j%50!=0:
                continue
            self.raw_data.append(data[j])
        print("We have %d annotated image" % (len(self.raw_data)))

    def get_cen(self, anno):
        center_rows = []
        center_cols = []
        for i in reversed(range(len(anno['lanes']))):
            cols = (np.array(anno['lanes'][i]).astype(float))
            rows = (np.array(anno['h_samples']).astype(float))
            mask = cols >= 0
            cols = cols[mask]
            rows = rows[mask]
            if rows.shape[0] == 0:
                del anno['lanes'][i]
                continue
            if rows[0] > rows[-1]:
                cols = cols[::-1]
            unique_idx = np.sort(np.unique(rows, return_index=True)[1])
            cols = cols[unique_idx]
            rows = rows[unique_idx]
            p1 = np.polyfit(rows, cols, 3)
            mask1 = 1280 > cols
            mask2 = cols >= 0
            mask3 = 720 > rows
            mask4 = rows >= 0
            cols[~mask1 | ~mask2| ~mask3| ~mask4] = -2
            filtered_cols = cols[cols != -2]
            filtered_rows = rows[cols != -2]
            if len(filtered_cols) == 0:
                print("No valid lane points remain after filtering while computing lane centers.")
                continue
            maxidx = np.argmax(filtered_rows)
            minidx = np.argmin(filtered_rows)
            center_row = 0.5 * (filtered_rows[minidx] + filtered_rows[maxidx])
            center_col = np.polyval(p1, center_row)
            center_rows.insert(0, center_row)
            center_cols.insert(0, center_col.item())
        center_point = list(zip(center_cols, center_rows))
        return center_point

    def get_data_org(self, idx):
        img = Image.open(self.root / self.raw_data[idx]["raw_file"]).convert('RGB')
        anno = self.raw_data[idx]
        cen = self.get_cen(anno)
        return img, anno, cen

    def get_data_aug(self, img, anno, cen):
        img_new, lane_new, cen_new = self.transform.process_for_test(img, anno, cen)
        img_new = Image.fromarray(img_new)
        img_new = self.to_tensor(img_new)
        return {'img': self.normalize(img_new),
                'lanes': lane_new,
                'cens': cen_new}

    def get_label_map(self, data):
        out = dict()
        self.label = dict()
        self.label['cen'] = np.zeros((cfg.Dataprocess_cfg.imgSize[0], cfg.Dataprocess_cfg.imgSize[1]))
        self.label['ins'] = np.zeros((cfg.Dataprocess_cfg.imgSize[0], cfg.Dataprocess_cfg.imgSize[1]))
        self.label['seg'] = np.zeros((cfg.Dataprocess_cfg.imgSize[0], cfg.Dataprocess_cfg.imgSize[1]))
        if not (len(data['lanes']) == len(data['cens'])):
            print("Inconsistent lane and center counts; returning empty labels.")
            out.update(self.label)
            return out
        laneidx=0
        for i in range(len(data['lanes'])):
            laneidx+=1
            lane_pts = data['lanes'][i].astype('int16')
            cen_pts = data['cens'][i]
            self.label['cen'] = cv2.circle(self.label['cen'], (int(cen_pts[0]), int(cen_pts[1])), color=[1, 1, 1], radius=1, thickness=-1)
            for num in range(0, len(lane_pts) - 1):
                if lane_pts[num][0] > 0 and lane_pts[num + 1][0] > 0:
                    cv2.line(self.label['seg'], (lane_pts[num][0], lane_pts[num][1]), (lane_pts[num + 1][0], lane_pts[num + 1][1]),
                             [1, 1, 1], cfg.Dataprocess_cfg.lane_width['seg'])
                    cv2.line(self.label['ins'], (lane_pts[num][0], lane_pts[num][1]), (lane_pts[num + 1][0], lane_pts[num + 1][1]),
                             [laneidx, laneidx, laneidx], cfg.Dataprocess_cfg.lane_width['ins'], cv2.LINE_AA)
        self.label['seg'] = np.int8(self.label['seg'] != 0)
        self.label['ins'] = np.int8(self.label['ins'])
        point = np.where(self.label['cen'] == 1)
        point = np.stack(point)
        for idx in range(0, point.shape[1]):
            self.label['cen'] = draw_gaussian(self.label['cen'], (point[1, idx], point[0, idx]), 10)
        out.update(self.label)
        return out

    def remove_dict_keys(self, data):
        keylist = ['lanes','cens']
        for key in keylist:
            data.pop(key)
        return data

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        out = dict()
        img, anno, cen = self.get_data_org(idx)
        out.update(self.get_data_aug(img, anno, cen))
        out.update(self.get_label_map(out))
        out = self.remove_dict_keys(out)
        return out['img'],out['ins'],out['seg'],out['cen']

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1, delte=6):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / delte)
    x, y = center
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    heatmap[y - top:y + bottom, x - left:x + right]=masked_heatmap
    return heatmap
