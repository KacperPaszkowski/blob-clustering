import math
import random
from turtle import color, distance
import numpy as np
import cv2
from tqdm import tqdm
import copy

class GraphClustering():

    class Cluster():
        def __init__(self, boxes):
            self.boxes = boxes
            self.centroid = self._cluster_centroid()

        def _cluster_centroid(self):
            positions = np.array([box.center for box in self.boxes])
            return [np.mean(positions[:, 0]), np.mean(positions[:, 1])]

        def merge_clusters(self, cluster):
            self.boxes = self.boxes + cluster.boxes
            self.centroid = self._cluster_centroid()

        def size(self):
            positions = np.array([box.center for box in self.boxes])
            return(np.min(positions[:,0]), np.min(positions[:,1]), np.max(positions[:,0]), np.max(positions[:,1]))


    def __init__(self, boxes):
        self.boxes = boxes
        self.clusters = []
        self._initialize_clusters()

        for x in range(5):
            self._distance_clustering()

    def _initialize_clusters(self):
        for box in self.boxes:
            self.clusters.append(self.Cluster([box]))

    def _distance(self, cord1, cord2):
        return np.linalg.norm(np.array(cord1) - np.array(cord2))
        
    def _distance_clustering(self):
        matched_clusters = {}
        for init_idx, x in enumerate(self.clusters.copy()):
            min_distance = math.inf
            min_cluster = None
            matched_id = None
            for idx, y in enumerate(self.clusters.copy()):
                distance = self._distance(x.centroid, y.centroid)
                matched = list(matched_clusters.keys()) + list(matched_clusters.values())
                if distance < min_distance and x != y and init_idx not in matched and idx not in matched:
                    min_distance = distance
                    min_cluster = y
                    matched_id = idx
            if min_cluster != None and matched_id != None:
                matched_clusters[init_idx] = matched_id

        new_clusters = []

        for match in matched_clusters:
            cluster_a = copy.deepcopy(self.clusters[match])
            cluster_b = copy.deepcopy(self.clusters[matched_clusters[match]])

            cluster_a.merge_clusters(cluster_b)
            new_clusters.append(cluster_a)
        
        self.clusters = copy.deepcopy(new_clusters)


class Box():
    def __init__(self, x, y, w, h, area):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.pos = (int(x - w/2), int(y - h/2), w, h)
        self.center = [x, y]
        self.area = area
        self.distance = []

class MaxPool():

    def __init__(self, image, scale):
        self.image = image
        self.scale = scale
        self.shape = None
        self.pooled = self.max_pool()


    def _add_padding(self):
        padded_x = self.image.shape[0] % self.scale
        padded_y = self.image.shape[1] % self.scale
        padded = np.zeros((self.image.shape[0] + self.scale - padded_x, self.image.shape[1] + self.scale - padded_y))
        padded[0:self.image.shape[0],0:self.image.shape[1]] = self.image
        return padded 

    def max_pool(self):
        padded = self._add_padding()
        x_scale = padded.shape[0] // self.scale
        y_scale = padded.shape[1] // self.scale
        self.shape = (x_scale, y_scale)
        out_image = np.zeros((x_scale, y_scale))
        for y in range(y_scale):
            for x in range(x_scale):
                out_image[x, y] = np.sum(padded[x*self.scale:(x+1)*self.scale, y*self.scale:(y+1)*self.scale]) > 0
                
        return out_image

    def translate_pos(self, x, y):
        return (x * self.scale + self.scale / 2, y * self.scale + self.scale / 2, self.scale, self.scale)

    def blob_area(self, x, y):
        x_c, y_c, w, h = self.translate_pos(x, y)
        x_t = int(x_c - w/2)
        y_t = int(y_c - h/2)
        return np.sum(self.image[y_t:y_t+h, x_t:x_t+w]) / (w*h)

class Detector():
    def load_image(self, path):
        self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.image[self.image > 0] = 1
        self.pools = []
        self.boxes = []
        return self.image

    def multiscale_detect(self, image, max_scale=600, min_scale=128, scale_factor=0.2):
        scale_factor_px = int((max_scale - min_scale) * scale_factor)
        for scale in range(max_scale, min_scale, -scale_factor_px):
            self.pools.append(MaxPool(image, scale))

        return self._draw_rects(image)


    def _draw_rects(self, image):
        new_img = np.zeros((image.shape[0], image.shape[1], 3))
        new_img[:,:,0] = image*255
        new_img[:,:,1] = image*255
        new_img[:,:,2] = image*255

        for pool in self.pools:
            for y in range(pool.shape[1]):
                for x in range(pool.shape[0]):
                    if pool.blob_area(x, y) > 0:
                        _x, _y, _w, _h = pool.translate_pos(x, y)
                        self.boxes.append(Box(_x, _y, _w, _h, pool.blob_area(x, y)))

        z = GraphClustering(self.boxes)

        for cluster in z.clusters:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            x, y = cluster.centroid
            x_1, y_1, x_2, y_2 = cluster.size()
            new_img = cv2.rectangle(new_img, (int(x), int(y)), (int(x+5), int(y+5)), color, 20)
            new_img = cv2.rectangle(new_img, (int(x_1), int(y_1)), (int(x_2), int(y_2)), color, 3)

        return new_img


    
x = Detector()
img = x.load_image("blobs.bmp")
cv2.imwrite('ok.bmp', x.multiscale_detect(img))
