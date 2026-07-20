import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmentables.kps import KeypointsOnImage, Keypoint

class Transforms(object):
    def __init__(self, inputW, inputH):
        self.width = inputW
        self.height = inputH

    def settings(self):
        transforms = self.custom_transforms(self.height, self.width)
        transforms_for_test = self.transforms_for_test(self.height, self.width)
        img_transforms = []
        for aug in transforms:
            p = aug['p']
            if aug['name'] != 'OneOf':
                img_transforms.append(
                    iaa.Sometimes(p=p, then_list=getattr(iaa, aug['name'])(**aug['parameters'])))
            else:
                img_transforms.append(
                    iaa.Sometimes(p=p, then_list=iaa.OneOf([getattr(iaa, aug_['name'])(**aug_['parameters']) for aug_ in aug['transforms']])))
        img_transforms_for_test = []
        for aug in transforms_for_test:
            p = aug['p']
            img_transforms_for_test.append(
                iaa.Sometimes(p=p, then_list=getattr(iaa, aug['name'])(**aug['parameters'])))
        self.transform = iaa.Sequential(img_transforms)
        self.transform_for_test = iaa.Sequential(img_transforms_for_test)

    def lane_to_linestrings(self, data):
        lane = list()
        for i in range(len(data)):
            pts = data[i]
            lane.append(LineString(pts))
        return lane

    def linestrings_to_lanes(self, data):
        lanes = []
        for pts in data:
            lanes.append(pts.coords)
        return lanes

    def cen_to_keypoints(self, center_row_coord, center_col_coord):
        keypoints = []
        for row, col in zip(center_row_coord, center_col_coord):
            keypoints.append(Keypoint(x=col, y=row))
        return keypoints

    def keypoints_to_cen(self, keypoints_on_image):
        keypoints_coords = keypoints_on_image.to_xy_array()
        center_points_list = keypoints_coords.tolist()
        return center_points_list

    def process(self, img_org, anno, cen):
        img_org = np.uint8(img_org)
        anno_pure = self.remove_minus_two(anno)
        line_strings_org = self.lane_to_linestrings(anno_pure)
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)
        center_row_coord = [pt[1] for pt in cen]
        center_col_coord = [pt[0] for pt in cen]
        cen_strings_org = self.cen_to_keypoints(center_row_coord, center_col_coord)
        cen_strings_org = KeypointsOnImage(cen_strings_org, shape=img_org.shape)
        img_new, line_strings_new, cen_strings_new = self.transform(image=img_org, line_strings=line_strings_org, keypoints=cen_strings_org)
        lane_new = self.linestrings_to_lanes(line_strings_new)
        cen_new = self.keypoints_to_cen(cen_strings_new)
        height, width = img_new.shape[:2]
        cen_new = [
            (min(max(x, 0), width - 1), min(max(y, 0), height - 1))
            for (x, y) in cen_new
        ]
        return img_new, lane_new, cen_new

    def remove_minus_two(self, annotation):
        new_coords = []
        for lane_coords in annotation['lanes']:
            arr = np.array(lane_coords, dtype=np.int32)
            has_minus2 = (arr == -2)
            arr_filtered = arr[~has_minus2]
            Y_filtered = np.array(annotation['h_samples'])[~has_minus2]
            lanes = np.stack([arr_filtered, Y_filtered], axis=1)
            new_coords.append(lanes)
        return new_coords

    def process_for_test(self, img_org, anno, cen):
        img_org = np.uint8(img_org)
        anno_pure = self.remove_minus_two(anno)
        line_strings_org = self.lane_to_linestrings(anno_pure)
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)
        center_row_coord = [pt[1] for pt in cen]
        center_col_coord = [pt[0] for pt in cen]
        cen_strings_org = self.cen_to_keypoints(center_row_coord, center_col_coord)
        cen_strings_org = KeypointsOnImage(cen_strings_org, shape=img_org.shape)
        img_new, line_strings_new, cen_strings_new = self.transform_for_test(image=img_org, line_strings=line_strings_org, keypoints=cen_strings_org)
        anno_new = self.linestrings_to_lanes(line_strings_new)
        cen_new = self.keypoints_to_cen(cen_strings_new)
        height, width = img_new.shape[:2]
        cen_new = [
            (min(max(x, 0), width - 1), min(max(y, 0), height - 1))
            for (x, y) in cen_new
        ]
        return img_new, anno_new, cen_new

    def custom_transforms(self, img_h, img_w):
        transform = [
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                 ],
                 p=0.2),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2)),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ]
        return transform

    def transforms_for_test(self, img_h, img_w):
        transform = [
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ]
        return transform
