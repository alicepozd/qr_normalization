import numpy as np
from matplotlib import pyplot as plt
import cv2
import json
import math
from cv2 import getPerspectiveTransform
from cv2 import warpPerspective
import math
import scipy.interpolate as interpolate


def getAngle(a, b, c): #  угол abc
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


def get_verteces(bounds): # возвращает углы qr кода
    X = bounds[0]
    Y = bounds[1]
    angles = [(0, getAngle((X[len(X) - 2], Y[len(X) - 2]), (X[0], Y[0]), (X[1], Y[1])))]
    for i in range(1, len(X) - 1):
        angles.append((i, getAngle((X[i - 1], Y[i - 1]), (X[i], Y[i]), (X[i + 1], Y[i + 1]))))
    verteces = [element[0] for element in sorted(angles, key=lambda x: x[1])[:4]]
    return sorted(verteces)


def preprocess_bounds(bounds): #  bounds = [X, Y], X = [u, r, d, l], u = []
    X = bounds[0]
    Y = bounds[1]
    verteces = get_verteces(bounds)
    coordinates = [(0, (X[verteces[0]], Y[verteces[0]])), (1, (X[verteces[1]], Y[verteces[1]])), (2, (X[verteces[2]], Y[verteces[2]])), (3, (X[verteces[3]], Y[verteces[3]]))]
    left_up = [element[0] for element in sorted(coordinates, key=lambda x: x[0], reverse=True)[:1]][0]
    verteces_shift = []
    for i in range(len(coordinates)):
        verteces_shift.append(verteces[(i+left_up)%4])
    bounds_preprocessed_X = [[], [], [], []]
    bounds_preprocessed_Y = [[], [], [], []]
    for i in range(len(bounds_preprocessed_X)):
        index = verteces_shift[i]
        while index != (verteces_shift[(i+1)%len(bounds_preprocessed_X)]+1)%(len(X)-1):
            bounds_preprocessed_X[i].append(X[index])
            bounds_preprocessed_Y[i].append(Y[index])
            index = (index+1)%(len(X)-1)
    return bounds_preprocessed_X, bounds_preprocessed_Y
    

def get_points_after_perspective(X, Y, reversed):
    if reversed:
        return [180, 20, 20, 180], [20, 20, 180, 180]
    return [20, 180, 180, 20], [180, 180, 20, 20]


def perspective_correction(image, raw_bounds): #  bounds = [X, Y], X = [u, r, d, l], u = []
    raw_bounds = preprocess_bounds(raw_bounds)
    pts1 = np.float32([[raw_bounds[0][0][0], raw_bounds[1][0][0]], [raw_bounds[0][1][0], raw_bounds[1][1][0]],
                    [raw_bounds[0][2][0], raw_bounds[1][2][0]], [raw_bounds[0][3][0], raw_bounds[1][3][0]]])
    pts2 = np.float32([[20, 180], [180, 180],
                        [180, 20], [20, 20]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(plt.imread(image), matrix, (200, 200))

    bounds = [[[], [], [], []], [[], [], [], []]]
    for i in range(4):
        for j in range(len(raw_bounds[0][i])):
            point = np.array([raw_bounds[0][i][j], raw_bounds[1][i][j], 1])
            new_point = matrix @ point
            bounds[0][i].append(new_point[0]/new_point[2])
            bounds[1][i].append(new_point[1]/new_point[2])

    return result, bounds
    
    
def get_curve_ind(X, Y): #  мера кривизны
    curve_ind = 0
    for i in range(len(X)):
        t = ((X[i] - X[0])*(X[len(X)-1] - X[0])+(Y[i] - Y[0])*(Y[len(X)-1] - Y[0]))/((X[len(X)-1] - X[0])**2 + (Y[len(X)-1] - Y[0])**2)
        curve_ind += ((X[0] - X[i] + (X[len(X)-1] - X[0])*t)**2 + (Y[0] - Y[i] + (Y[len(X)-1] - Y[0])*t)**2)**0.5
    return curve_ind/len(X)


def find_non_linear(X, Y):
    max_curve_ind = -1
    index = -1
    for i in range(len(X)):
        curve_ind = get_curve_ind(X[i], Y[i])
        if index == -1 or curve_ind > max_curve_ind:
            index = i
            max_curve_ind = curve_ind
    return index, max_curve_ind


def curve_to_up(image, bounds):
    non_linear_ind = find_non_linear(bounds[0], bounds[1])
    if non_linear_ind[0]%2 != 0:
        return np.transpose(image, axes=(1, 0, 2)), (bounds[1], bounds[0]), 1
    return image, (bounds[0], bounds[1]), 0
    
import scipy.interpolate as interpolate


class FuncRemap:
    def __init__(self, bounds, is_rotated):
        if is_rotated:
            X1, Y1 = bounds[0][is_rotated][::-1], bounds[1][is_rotated][::-1]
            X2, Y2 = bounds[0][is_rotated+2], bounds[1][is_rotated+2]
        else:
            X1, Y1 = bounds[0][is_rotated], bounds[1][is_rotated]
            X2, Y2 = bounds[0][is_rotated+2][::-1], bounds[1][is_rotated+2][::-1]
        if len(X1) < 4:
            self.lflg1 = 1
            self.y1 = Y1[0]
        else:
            self.lflg1 = 0
            self.poly1 = interpolate.splrep(X1, Y1)
        if len(X2) < 4:
            self.lflg2 = 1
            self.y2 = Y2[0]
        else:
            self.lflg2 = 0
            self.poly2 = interpolate.splrep(X2, Y2)

    def x(self, x, y):
        return x

    def y(self, x, y):
        t = (y - 20)/160
        if self.lflg1:
            y1 = self.y1
        else:
            y1 = interpolate.splev(x, self.poly1)
        if self.lflg2:
            y2 = self.y2
        else:
            y2 = interpolate.splev(x, self.poly2)
        ans = (t*y1 + (1-t)*y2)
        return min(199, max(0,ans))


def make_linear(image, bounds):
    persp_corr_i, persp_corr_b = perspective_correction(image, bounds)
    rot_i, rot_b, up_ind = curve_to_up(persp_corr_i, persp_corr_b)
    res = np.zeros_like(rot_i)
    remapper = FuncRemap(rot_b, up_ind)
    for x in np.arange(0, 200):
        for y in np.arange(0, 200):
            res[y][x] = rot_i[round(remapper.y(x, y))][round(remapper.x(x, y))]
    return res


def parse_data(dataset_path):
    images = []
    bounds = []
    images_names = []

    with open(dataset_path + '/train/_annotations.coco.json') as f:
        templates = json.load(f)

    for section, commands in templates.items():
        if section == 'images':
            for i in range(len(commands)):
                images.append(dataset_path + '/train/' + commands[i]['file_name'])
                images_names.append(commands[i]['file_name'])
        if section == 'annotations':
            for i in range(len(commands)):
                bounds.append(commands[i]['segmentation'])
    return images, bounds, images_names


def get_statistics(dataset_path):
    images, bounds, images_names = parse_data(dataset_path)
    qcd = cv2.QRCodeDetector()
    
    dataset_size, no_norm_stat, norm_stat, good_cases, bad_cases = 0, 0, 0, 0, 0
    for image_i in range(len(images)):
        try:
            X = bounds[image_i][0][::2]
            Y = bounds[image_i][0][1::2]
            t_retval_no_norm, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(plt.imread(images[image_i]))
            
            image_c = make_linear(images[image_i], (X, Y))
            plt.savefig('normalization_results/'+ images_names[image_i])
            retval_norm, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(image_c)

            dataset_size += 1
            if (t_retval_no_norm):
                no_norm_stat += 1
            if (retval_norm):
                norm_stat += 1
            if t_retval_no_norm == 0 and retval_norm == 1:
                good_cases += 1
            if t_retval_no_norm == 1 and retval_norm == 0:
                bad_cases += 1
            if t_retval_no_norm == 0 and retval_norm == 0:
                print('Have not been detected in any case: image', images_names[image_i])
        except:
            continue

    print()
    print('Dataset size: ', dataset_size)
    print('Decoded without normalization: ', no_norm_stat)
    print('Decoded with normalization: ', norm_stat)
    print('Normalization additionaly decoded: ', good_cases)
    print('Normalizations errors for QR that can be decoded: ', bad_cases)
    return dataset_size, no_norm_stat, norm_stat, dataset_size, bad_cases    
    
