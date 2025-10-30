import random
import cv2
import numpy as np
import math
import torch


class DualCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class NoTransform(object):
    def __call__(self, *args):
        return args[0]


class ToTensor(object):
    def __call__(self, *args):
        tmp = []
        for e in args:
            if e is not None:
                tmp.append(torch.from_numpy(e))
            else:
                tmp.append(None)
        # return [torch.from_numpy(e) for e in args]
        return tmp


# class Normalize:
#     def __init__(self, mean, std):
#         self.mean = np.array(mean)
#         self.std = np.array(std)
#
#     def __call__(self, img, mask=None):
#         img = img.astype(np.float32)
#         img = (img - self.mean) / self.std
#
#         import cv2
#         # cv2.imwrite('./sessions/3.png', img[0, :, :] * 100)
#         return img


class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img, mask=None):
        img = img.astype(np.float32)

        for c in range(3):
            img[c] = (img[c] - self.mean[c]) / self.std[c]

        img = torch.from_numpy(img).float()
        return img


class OneOf(object):
    def __init__(self, transforms, prob=.5):
        self.transforms = transforms
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img):
        if self.state['p'] < self.prob:
            img = self.state['t'](img)
        return img

    def randomize(self):
        self.state['p'] = random.random()
        self.state['t'] = random.choice(self.transforms)
        self.state['t'].prob = 1.


class Crop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, tuple):
            self.output_size = output_size
        else:
            raise ValueError('Incorrect value')
        # self.keep_size = keep_size
        # self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img):
        rows_in, cols_in = img.shape[1:]
        rows_out, cols_out = self.output_size
        rows_out = min(rows_in, rows_out)
        cols_out = min(cols_in, cols_out)

        r0 = math.floor(self.state['r0f'] * (rows_in - rows_out))
        c0 = math.floor(self.state['c0f'] * (cols_in - cols_out))
        r1 = r0 + rows_out
        c1 = c0 + cols_out
        # cv2.imwrite('./sessions/05.png', img[0, :, :])

        img = np.ascontiguousarray(img[:, r0:r1, c0:c1])
        # cv2.imwrite('./sessions/5.png', img[0, :, :])
        return img

    def randomize(self):
        # self.state['p'] = random.random()
        self.state['r0f'] = random.random()
        self.state['c0f'] = random.random()


class Scale(object):
    def __init__(self, ratio_range=(0.7, 1.2), prob=.5):
        self.ratio_range = ratio_range
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img):
        """
        Parameters
        ----------
        img: (ch, d0, d1) ndarray
        """
        if self.state['p'] < self.prob:
            ch, d0_i, d1_i = img.shape
            d0_o = math.floor(d0_i * self.state['r'])
            d0_o = d0_o + d0_o % 2
            d1_o = math.floor(d1_i * self.state['r'])
            d1_o = d1_o + d1_o % 2

            # img1 = cv2.copyMakeBorder(img, limit, limit, limit, limit,
            #                           borderType=cv2.BORDER_REFLECT_101)
            # img = np.squeeze(img)

            img0 = cv2.resize(img[0, :, :], (d1_o, d0_o))
            img1 = cv2.resize(img[1, :, :], (d1_o, d0_o))
            img2 = cv2.resize(img[2, :, :], (d1_o, d0_o))
            img_final = np.empty((3, *(d0_o, d1_o)), dtype=img.dtype)
            img_final[0, :, :] = img0
            img_final[1, :, :] = img1
            img_final[2, :, :] = img2


            # img = cv2.resize(img, (d1_o, d0_o), interpolation=cv2.INTER_LINEAR)
            # img = img[None, ...]
            # print('end', img_final.shape)
            # cv2.imwrite('./sessions/4.png', img[0, :, :])
            return img_final
        else:
            return img

    def randomize(self):
        self.state['p'] = random.random()
        self.state['r'] = round(random.uniform(*self.ratio_range), 2)


class pcld_rotate(object):
    def __init__(self, degree_range=(-5, 5), prob=0.5):
        self.theta_range = torch.deg2rad(torch.Tensor(degree_range))
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, pcld):
        """
        Parameters
        ----------
        point cloud: (pt, 3) 2D Tensor
        """

        if self.state["p"] < self.prob:
            cos_angle = np.cos(self.theta_range)
            sin_angle = np.sin(self.theta_range)

            rotation_matrix = np.array([
                [cos_angle, 0, sin_angle],
                [0, 1, 0],
                [-sin_angle, 0, cos_angle]
            ])

            rotated_pcld = pcld.dot(rotation_matrix.T)

            return rotated_pcld

        else:
            return pcld

    def randomize(self):
        self.state["p"] = random.random()
        self.state["theta"] = random.uniform(*self.theta_range)


class Rotate(object):
    def __init__(self, degree_range=(-30., 30.), prob=0.5):
        self.theta_range = torch.deg2rad(torch.Tensor(degree_range))
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, image):
        """
        Parameters
        ----------
        image: (CH, R, C, S) 4D Tensor
        """

        if self.state["p"] < self.prob:
            center_pt_x = int((image.shape[1]) / 2)
            center_pt_y = int((image.shape[2]) / 2)
            center_pt = np.array([center_pt_x, center_pt_y])

            M = cv2.getRotationMatrix2D((center_pt_x, center_pt_y), self.state["theta"].item(), scale=1)
            M[0, 2] = center_pt[0] / image.shape[2]
            M[1, 2] = center_pt[1] / image.shape[1]

            (h, w) = (image.shape[1], image.shape[2])

            ret_tmp = image.copy()
            ret0 = ret_tmp[0, :, :]
            ret1 = ret_tmp[1, :, :]
            ret2 = ret_tmp[2, :, :]

            ret0 = cv2.warpAffine(ret0, M, (w, h))
            ret1 = cv2.warpAffine(ret1, M, (w, h))
            ret2 = cv2.warpAffine(ret2, M, (w, h))

            ret_final = np.empty((3, *(h, w)), dtype=ret_tmp.dtype)
            ret_final[0, :, :] = ret0
            ret_final[1, :, :] = ret1
            ret_final[2, :, :] = ret2

            return ret_final

        else:
            return image

    def randomize(self):
        self.state["p"] = random.random()
        self.state["theta"] = random.uniform(*self.theta_range)


class MoveObject(object):
    def __init__(self, max_dx=20, max_dy=20, p=0.5, background_value=0):
        self.max_dx = max_dx
        self.max_dy = max_dy
        self.p = p
        self.background_value = background_value

    def __call__(self, img):
        """
        Parameters
        ----------
        img: (CH, R, C, S) 4D Tensor
        """

        if random.random() > self.p:
            return img

        c, h, w = img.shape

        mask = (img != self.background_value).any(axis=0)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        obj = img[:, y_min:y_max+1, x_min:x_max+1]

        max_up = max(0, y_min - self.max_dy)
        max_down = min(h - (y_max - y_min + 1), y_min + self.max_dy)
        max_left = max(0, x_min - self.max_dx)
        max_right = min(w - (x_max - x_min + 1), x_min + self.max_dx)

        new_y = random.randint(max_up, max_down)
        new_x = random.randint(max_left, max_right)

        new_img = np.ones_like(img) * self.background_value
        new_img[:, new_y:new_y+obj.shape[1], new_x:new_x+obj.shape[2]] = obj

        # cv2.imwrite('sessions/0.png', img[0, :, :])
        # cv2.imwrite('sessions/1.png', new_img[0, :, :])

        return new_img


class HorizontalFlip(object):
    def __init__(self, prob=.5):
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img):
        """
        Parameters
        ----------
        img: (ch, d0, d1) ndarray
        """
        if self.state['p'] < self.prob:
            img = np.flip(img, axis=1)
        return img

    def randomize(self):
        self.state['p'] = random.random()


class VerticalFlip(object):
    def __init__(self, prob=.5):
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img):
        """
        Parameters
        ----------
        img: (ch, d0, d1) ndarray
        """
        if self.state['p'] < self.prob:
            img = np.flip(img, axis=0)
        return img

    def randomize(self):
        self.state['p'] = random.random()


# class CutOutWrapper:
#     def __init__(self, batch_size=5, prob=0.5):
#         self.cutout = CustomCutOut(batch_size=batch_size)
#         self.prob = prob
#
#     def __call__(self, img):
#         if np.random.rand() > self.prob:
#             return img
#         img_tensor = torch.from_numpy(img)
#
#         # Randomize cutout mask
#         self.cutout.randomize()
#
#         # Apply cutout
#         img_tensor = self.cutout(img_tensor)
#         a = img_tensor.numpy()
#
#         return a


class CustomCutOut:
    def __init__(self, holes=1, hole_size=(60, 60), fill_value=0, prob=0.5):
        self.holes = holes
        self.hole_size = hole_size
        self.fill_value = fill_value
        self.prob = prob

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        img = img.copy()
        c, h, w = img.shape
        for _ in range(self.holes):
            y = np.random.randint(0, h - self.hole_size[0])
            x = np.random.randint(0, w - self.hole_size[1])
            img[:, y:y+self.hole_size[0], x:x+self.hole_size[1]] = self.fill_value
        return img
