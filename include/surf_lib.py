import numpy as np
import cv2
from functools import lru_cache
import cloudpickle


class SURF:
    __octaves: int = 4
    __layers: int = 4
    __threshold: int = 1000
    __image: np.ndarray
    __image_stretched: np.ndarray
    __border_size: int = 0
    __integral_image: np.ndarray
    __box_space_params: dict
    __L_by_octaves_layers: dict

    __DetHes: np.ndarray  # determinants of Hessian
    __signLaplassian: np.ndarray  # determinants of Hessian

    def __init__(self, octaves=4, layers=4, threshold=1000):
        self.__octaves = octaves
        self.__layers = layers
        self.__threshold = threshold
        self.__box_space_params = {}
        self.__L_by_octaves_layers = {}
        self.__compute_box_space_parameters()

    def init(self, gray_image: np.ndarray):
        self.__image = gray_image
        self.__build_stretched_image()
        self.__build_integral_image()
        self.__compute_determinant_of_hessian()

    def detectAndCompute(self):
        """
        Returns list of keypoints cv2.KeyPoint and SURF descriptors
        Each descriptor is a list with 3 elements:
         - (64,) descriptor vector
         - orientation
         - sign of Laplassian
        """
        key_points = self.__keypoints_selection()

        cv2_key_points = []
        descriptors = []

        for key_point in key_points:
            x, y, L, orientation = key_point
            point = cv2.KeyPoint(
                x=x,
                y=y,
                size=self.__get_box_space_parameter_by_L(L, 'sigma_L'),
                angle=orientation * 180 / np.pi
            )
            cv2_key_points.append(point)

            descriptor = self.__compute_point_descriptor(key_point)
            descriptors.append(list(descriptor))

        return cv2_key_points, descriptors

    def __build_integral_image(self):
        self.__integral_image = np.cumsum(np.cumsum(self.__image_stretched, axis=0), axis=1)

    def __build_stretched_image(self):
        """
        Original image will be extended with border size n around
        and this border will be filled by mirror of the original image
        """
        n = self.__border_size
        h = self.__image.shape[0]
        w = self.__image.shape[1]

        self.__image_stretched = np.ndarray((h + 2 * n, w + 2 * n))
        self.__image_stretched[n:n + h, n:n + w] = self.__image
        self.__image_stretched[0:n, n:n + w] = np.flipud(self.__image[0:n, :])
        self.__image_stretched[n + h:, n:n + w:] = np.flipud(self.__image[h - n:, :])
        self.__image_stretched[:, 0:n] = np.fliplr(self.__image_stretched[:, n:2 * n])
        self.__image_stretched[:, -n:] = np.fliplr(self.__image_stretched[:, -2 * n:-n])

    def __compute_box_space_parameters(self):
        self.__get_box_space_parameter_by_L.cache_clear()
        self.__get_Dxx_L_domains.cache_clear()
        self.__get_Dyy_L_domains.cache_clear()
        self.__get_Dxy_L_domains.cache_clear()
        self.__get_L_index_by_L.cache_clear()
        self.__get_L_by_L_index.cache_clear()

        for o in range(1, self.__octaves + 1):
            self.__L_by_octaves_layers[o] = {}
            for i in range(1, self.__layers + 1):
                L = (2 ** o) * i + 1
                self.__L_by_octaves_layers[o][i] = L
                self.__box_space_params[L] = {
                    'sigma_L': round(0.4 * L, 2),
                    'L': L,
                    'l': np.int(0.8 * L),
                    'w': np.sqrt((2 * L - 1) / (2 * L))
                }

        L_max = 3 * ((2 ** self.__octaves) * self.__layers + 1)
        self.__border_size = L_max

    @lru_cache
    def __get_box_space_parameter_by_L(self, L: int, param: str):
        return self.__box_space_params[L][param]

    def __box_convolution(self, a: int, b: int, c: int, d: int, x: int, y: int):
        """
        Consider params as rectangular domain: [a,b]x[c,d] - not points!!
        and point (x,y)
        """

        xa = self.__border_size + x - a - 1
        yc = self.__border_size + y - c - 1
        xb = self.__border_size + x - b - 1 - 1
        yd = self.__border_size + y - d - 1 - 1

        if (0 <= xa < self.__integral_image.shape[1]) and (0 < yc < self.__integral_image.shape[0]):
            u1 = self.__integral_image[yc, xa]
        else:
            u1 = 0

        if (0 <= xb < self.__integral_image.shape[1]) and (0 < yd < self.__integral_image.shape[0]):
            u2 = self.__integral_image[yd, xb]
        else:
            u2 = 0

        if (0 <= xa < self.__integral_image.shape[1]) and (0 < yd < self.__integral_image.shape[0]):
            u3 = self.__integral_image[yd, xa]
        else:
            u3 = 0

        if (0 <= xb < self.__integral_image.shape[1]) and (0 < yc < self.__integral_image.shape[0]):
            u4 = self.__integral_image[yc, xb]
        else:
            u4 = 0

        return u1 + u2 - u3 - u4

    def __get_image_value(self, x: int, y: int):
        x_is = self.__border_size + x
        y_is = self.__border_size + y

        return self.__image_stretched[y_is][x_is]

    def __compute_Dx_L_convolution(self, l: int, x: int, y: int):
        return self.__box_convolution(-l, -1, -l, l, x, y) - self.__box_convolution(1, l, -l, l, x, y)

    def __compute_Dy_L_convolution(self, l: int, x: int, y: int):
        return self.__box_convolution(-l, l, -l, -1, x, y) - self.__box_convolution(-l, l, 1, l, x, y)

    def __compute_Dxx_L_convolution(self, L: int, x: int, y: int):
        (b1, b2), (b3, b4), (b5, b6), (b7, b8) = self.__get_Dxx_L_domains(L)
        B1 = self.__box_convolution(b1, b2, b3, b4, x, y)
        B2 = self.__box_convolution(b5, b6, b7, b8, x, y)
        return B1 - 3 * B2

    def __compute_Dyy_L_convolution(self, L: int, x: int, y: int):
        (b1, b2), (b3, b4), (b5, b6), (b7, b8) = self.__get_Dyy_L_domains(L)
        B1 = self.__box_convolution(b1, b2, b3, b4, x, y)
        B2 = self.__box_convolution(b5, b6, b7, b8, x, y)
        return B1 - 3 * B2

    def __compute_Dxy_L_convolution(self, L: int, x: int, y: int):
        (ne1, ne2), (ne3, ne4), \
            (nw1, nw2), (nw3, nw4), \
            (sw1, sw2), (sw3, sw4), \
            (se1, se2), (se3, se4) \
            = self.__get_Dxy_L_domains(L)

        north_east_quadrant = self.__box_convolution(ne1, ne2, ne3, ne4, x, y)
        north_west_quadrant = self.__box_convolution(nw1, nw2, nw3, nw4, x, y)
        south_west_quadrant = self.__box_convolution(sw1, sw2, sw3, sw4, x, y)
        south_east_quadrant = self.__box_convolution(se1, se2, se3, se4, x, y)

        return north_east_quadrant + south_west_quadrant - north_west_quadrant - south_east_quadrant

    @lru_cache
    def __get_Dxx_L_domains(self, L: int):
        b1 = np.int((L * 3 - 1) / 2)
        b2 = np.int((L - 1) / 2)
        return [
            (-b1, b1),
            (-(L - 1), (L - 1)),
            (-b2, b2),
            (-(L - 1), (L - 1))
        ]

    @lru_cache
    def __get_Dyy_L_domains(self, L: int):
        b1 = np.int((L * 3 - 1) / 2)
        b2 = np.int((L - 1) / 2)
        return [
            (-(L - 1), (L - 1)),
            (-b1, b1),
            (-(L - 1), (L - 1)),
            (-b2, b2)
        ]

    @lru_cache
    def __get_Dxy_L_domains(self, L: int):
        return [
            (1, L), (1, L),  # ++
            (-L, -1), (1, L),  # -+
            (-L, -1), (-L, -1),  # --
            (1, L), (-L, -1),  # +-
        ]

    @lru_cache
    def __get_L_index_by_L(self, L: int):
        if L not in self.__box_space_params:
            return False

        return list(self.__box_space_params).index(L)

    @lru_cache
    def __get_L_by_L_index(self, L_index: int):
        if 0 <= L_index < len(self.__box_space_params):
            list_L = list(self.__box_space_params)
            return list_L[L_index]

        return False

    def __compute_determinant_of_hessian(self):
        self.__getDH.cache_clear()
        self.__getSignLaplassian.cache_clear()

        used_L = set({})
        self.__DetHes = np.ndarray(
            (len(self.__box_space_params),
             self.__image.shape[0],
             self.__image.shape[1])
        )
        self.__signLaplassian = np.ndarray(
            (len(self.__box_space_params),
             self.__image.shape[0],
             self.__image.shape[1])
        )
        DoH_index = 0

        for o in range(1, self.__octaves + 1):
            step = 2 ** (o - 1)

            for i in range(1, self.__layers + 1):
                L = (2 ** o) * i + 1

                if L in used_L:
                    continue

                used_L.add(L)

                normalizer = 1 / (L ** 4)
                w = self.__get_box_space_parameter_by_L(L, 'w')
                x = 0

                DoH_L = np.ndarray(self.__image.shape)
                signLaplassian_L = np.ndarray(self.__image.shape)

                while x < self.__image.shape[1]:
                    y = 0
                    while y < self.__image.shape[0]:
                        Dxx = self.__compute_Dxx_L_convolution(L, x, y)
                        Dyy = self.__compute_Dyy_L_convolution(L, x, y)
                        Dxy = self.__compute_Dxy_L_convolution(L, x, y)

                        DoH = normalizer * (Dxx * Dyy - ((w * Dxy) ** 2))
                        DoH_L[y][x] = DoH

                        signLaplassian_L[y][x] = 1 if (Dxx + Dyy) > 0 else -1

                        y += step

                    x += step

                self.__DetHes[DoH_index] = DoH_L
                self.__signLaplassian[DoH_index] = signLaplassian_L
                DoH_index += 1

    @lru_cache
    def __getDH(self, L: int, x: int, y: int):
        exist = (0 <= L < self.__DetHes.shape[0]) \
                and (0 <= x < self.__DetHes.shape[2]) \
                and (0 <= y < self.__DetHes.shape[1])

        if not exist:
            return -np.inf

        val = self.__DetHes[L][y][x]

        return val

    @lru_cache
    def __getSignLaplassian(self, L: int, x: int, y: int):
        exist = (0 <= L < self.__signLaplassian.shape[0]) \
                and (0 <= x < self.__signLaplassian.shape[2]) \
                and (0 <= y < self.__signLaplassian.shape[1])

        if not exist:
            return 0

        val = self.__signLaplassian[L][y][x]

        return val

    def __keypoints_selection(self):
        keyPoints = []
        reffal = 0

        for o in range(1, self.__octaves + 1):
            step = 2 ** (o - 1)

            for i in range(2, np.min([self.__layers, 4])):
                L = (2 ** o) * i + 1
                L_index = self.__get_L_index_by_L(L)
                x = 0

                while x < self.__image.shape[1]:
                    y = 0
                    while y < self.__image.shape[0]:
                        DoH = self.__getDH(L_index, x, y)

                        if DoH > self.__threshold:
                            if self.__is_DoH_local_maximum(L_index, x, y, DoH):
                                isRefinedKeyPoint, point = self.__is_refined(x, y, L, o)

                                if isRefinedKeyPoint and point[2] is False:
                                    reffal += 1
                                if isRefinedKeyPoint and point[2] is not False:
                                    key_point_with_orientation = self.__compute_key_point_orientation(point)
                                    keyPoints.append(key_point_with_orientation)

                        y += step

                    x += step

        # print(f'Refined and false L {reffal}')

        return keyPoints

    def __is_DoH_local_maximum(self, L, x, y, current_DoH):
        lL = L - 1 if L - 1 >= 0 else 0
        lx = x - 1 if x - 1 >= 0 else 0
        ly = y - 1 if y - 1 >= 0 else 0

        try:
            max_value = np.amax(self.__DetHes[lL:L + 2, ly:y + 2, lx:x + 2])
        except BaseException:
            max_value = np.inf
            pass

        if max_value > current_DoH:
            return False

        return True

    def __is_refined(self, x0, y0, L, o):
        p = 2 ** (o - 1)
        p2 = p ** 2
        L0 = self.__get_L_index_by_L(L)
        Lp2p = self.__get_L_index_by_L(L + 2 * p)
        Lm2p = self.__get_L_index_by_L(L - 2 * p)

        Hxx = (self.__getDH(L0, x0 + p, y0) + self.__getDH(L0, x0 - p, y0)
               - 2 * self.__getDH(L0, x0, y0)) / p2

        Hyy = (self.__getDH(L0, x0, y0 + p) + self.__getDH(L0, x0, y0 - p)
               - 2 * self.__getDH(L0, x0, y0)) / p2

        Hxy = (self.__getDH(L0, x0 + p, y0 + p) + self.__getDH(L0, x0 - p, y0 - p)
               - self.__getDH(L0, x0 - p, y0 + p) - self.__getDH(L0, x0 + p, y0 - p)) / (4 * p2)

        HxL = (self.__getDH(Lp2p, x0 + p, y0) + self.__getDH(Lm2p, x0 - p, y0)
               - self.__getDH(Lp2p, x0 - p, y0) - self.__getDH(Lm2p, x0 + p, y0)) / (8 * p2)

        HyL = (self.__getDH(Lp2p, x0, y0 + p) + self.__getDH(Lm2p, x0, y0 - p)
               - self.__getDH(Lp2p, x0, y0 - p) - self.__getDH(Lm2p, x0, y0 + p)) / (8 * p2)

        HLL = (self.__getDH(Lp2p, x0, y0) + self.__getDH(Lm2p, x0, y0) - 2 * self.__getDH(L0, x0, y0)) / (4 * p2)

        dx = (self.__getDH(L0, x0 + p, y0) - self.__getDH(L0, x0 - p, y0)) / (2 * p)
        dy = (self.__getDH(L0, x0, y0 + p) - self.__getDH(L0, x0, y0 - p)) / (2 * p)
        dL = (self.__getDH(Lp2p, x0, y0) - self.__getDH(Lm2p, x0, y0)) / (4 * p)

        # build matrix H0
        H0 = np.zeros((3, 3))
        np.fill_diagonal(H0, [Hxx, Hyy, HLL])
        H0[0, 1] = H0[1, 0] = Hxy
        H0[0, 2] = H0[2, 0] = HxL
        H0[1, 2] = H0[2, 1] = HyL

        # build vector d0
        d0 = np.array([dx, dy, dL])

        # compute ksi
        try:
            ksi = np.dot(np.linalg.inv(H0), d0)

            if np.max(np.abs(ksi * [1, 1, 0.5])) < p:
                x = np.int(x0 + ksi[0])
                y = np.int(y0 + ksi[1])
                L_index = np.int(L0 + ksi[2])

                return True, (x, y, self.__get_L_by_L_index(L_index))
        except np.linalg.LinAlgError:
            pass

        return False, (-1, -1, -1)

    @lru_cache
    def __get_gaussian(self, x, y, sigma):
        s2 = 2 * sigma * sigma
        return np.exp(-(x * x + y * y) / s2) / (np.pi * s2)

    def __compute_key_point_orientation(self, key_point):
        x0, y0, L = key_point
        l = self.__get_box_space_parameter_by_L(L, 'l')
        sigma = self.__get_box_space_parameter_by_L(L, 'sigma_L')
        sectors = 20

        list_fi = np.ndarray((0, 3))
        list_theta = np.ndarray((0, 4))
        G_sum = 0
        G_list = {}

        for i in range(-6, 7):
            for j in range(-6, 7):
                if i * i + j * j <= 36:
                    x = np.int(x0 + i * sigma)
                    y = np.int(y0 + j * sigma)
                    G = self.__get_gaussian(x - x0, y - y0, 2 * sigma)
                    G_sum += G
                    G_list.setdefault(10 * i + j, G)

        for i in range(-6, 7):
            for j in range(-6, 7):
                if i * i + j * j <= 36:
                    x = np.int(x0 + i * sigma)
                    y = np.int(y0 + j * sigma)
                    DxL = self.__compute_Dx_L_convolution(l, x, y)
                    DyL = self.__compute_Dy_L_convolution(l, x, y)
                    G = G_list.get(10 * i + j) / G_sum
                    image_value = self.__get_image_value(x, y)

                    fi = np.array([[
                        np.arctan2(DyL, DxL),
                        DxL * image_value * G,
                        DyL * image_value * G,
                    ]])

                    list_fi = np.concatenate([list_fi, fi])

        for k in range(0, sectors * 2):
            theta = k * np.pi / sectors

            mask = (theta - np.pi / 6 <= list_fi[:, 0]) & (list_fi[:, 0] <= theta + np.pi / 6)
            FI_theta = np.sum(list_fi[mask, 1:], axis=0)
            FI_theta_norm = np.linalg.norm(FI_theta)

            list_theta = np.concatenate([
                list_theta,
                np.concatenate([[theta], [FI_theta_norm], FI_theta]).reshape(1, 4)
            ])

        max_theta_norm_index = np.argmax(list_theta[:, 1])
        orientation = np.arctan2(list_theta[max_theta_norm_index][3], list_theta[max_theta_norm_index][2])

        return x0, y0, L, orientation

    def __compute_point_descriptor(self, point):
        x0, y0, L, theta = point
        Dl = self.__get_box_space_parameter_by_L(L, 'l')
        sigma = self.__get_box_space_parameter_by_L(L, 'sigma_L')
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        R_minus = np.array([
            [np.cos(-theta), -np.sin(-theta)],
            [np.sin(-theta), np.cos(-theta)],
        ])

        G_sum = 0
        G_list = {}
        step = 1

        # compute gausians to be able to get normalized discrete gausian
        for i in range(0, 4):
            for j in range(0, 4):
                for k in range(0, 5):
                    for l in range(0, 5):
                        u = (i - 2) * 5 + k + 0.5
                        v = (j - 2) * 5 + l + 0.5

                        sd = 3.3 * sigma
                        G = self.__get_gaussian(k, l, sd)
                        G_sum += G
                        G_list.setdefault(1000 * k + l, G)

        mk = np.zeros((4, 4, 4))

        for i in range(0, 4):
            for j in range(0, 4):
                sum_dx_uv = 0
                sum_dy_uv = 0
                sum_dx_uv_abs = 0
                sum_dy_uv_abs = 0

                for k in range(0, 5):
                    for l in range(0, 5):
                        u = (i - 2) * 5 + k + 0.5
                        v = (j - 2) * 5 + l + 0.5

                        x, y = sigma * (R @ np.array([u, v])) + np.array([x0, y0])
                        x = np.int(x)
                        y = np.int(y)

                        try:
                            DxL = self.__compute_Dx_L_convolution(Dl, x, y)
                            DyL = self.__compute_Dy_L_convolution(Dl, x, y)
                        except BaseException as e:
                            # print(x0, y0, L, theta, e)
                            pass

                        G = G_list.get(1000 * k + l) / G_sum

                        dx_uv, dy_uv = (R_minus @ np.array([DxL, DyL])) * G

                        sum_dx_uv += dx_uv
                        sum_dy_uv += dy_uv
                        sum_dx_uv_abs += np.abs(dx_uv)
                        sum_dy_uv_abs += np.abs(dy_uv)

                mk[i, j, 0] += sum_dx_uv
                mk[i, j, 1] += sum_dy_uv
                mk[i, j, 2] += sum_dx_uv_abs
                mk[i, j, 3] += sum_dy_uv_abs

        mk = mk.reshape((64,))

        norm = np.linalg.norm(mk)

        if norm != 0:
            mk = mk / norm

        return mk, theta, self.__getSignLaplassian(L, x0, y0)

class SURF_Matcher():
    def match(self, descriptors1, descriptors2, threshold=0.75):
        """
        Returns list of descriptors [cv2.DMatch]
        According to SURF match logic only distance between descriptor vectors with
        the same sign of Laplassian are considered
        """
        matches = []

        v1_index = 0
        for desc_vector1, theta1, SignLaplassian1 in descriptors1:
            lengths_to_point = np.ndarray((len(descriptors2),))
            v2_index = 0

            for desc_vector2, theta2, SignLaplassian2 in descriptors2:
                dist = np.inf

                if (SignLaplassian1 == SignLaplassian2):
                    dist = np.linalg.norm(desc_vector1 - desc_vector2)

                lengths_to_point[v2_index] = dist
                v2_index += 1

            indxs_sorted = np.argpartition(lengths_to_point, 2)
            dist1 = lengths_to_point[indxs_sorted[0]]
            dist2 = lengths_to_point[indxs_sorted[1]]

            if dist2 == 0 or dist1 / dist2 <= threshold:
                match = cv2.DMatch(
                    _distance=dist1,
                    _queryIdx=v1_index,
                    _trainIdx=indxs_sorted[0]
                )

                matches.append(match)

            v1_index += 1

        return matches


def save_keypoints_to_file(keyPoints, path):
    index = []
    for point in keyPoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        index.append(temp)

    cloudpickle.dump(index, open(path, 'wb'))


def load_keypoints_from_file(path):
    index = cloudpickle.load(open(path, 'rb'))
    keyPoints = []
    for point in index:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], size=point[1], angle=point[2], response=point[3],
                            octave=point[4], class_id=point[5])
        keyPoints.append(temp)

    return keyPoints


def save_descriptors_to_file(descriptors, path):
    cloudpickle.dump(descriptors, open(path, 'wb'))


def load_descriptors_from_file(path):
    descriptors = cloudpickle.load(open(path, 'rb'))

    return descriptors
