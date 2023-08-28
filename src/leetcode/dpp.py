import numpy as np
import math
import random
from collections import OrderedDict


class Item:
    def __init__(self, id, r_score2, embedding):
        self.id = id
        self.r_score = r_score2
        self.d_score = r_score2
        self.embedding = embedding
        self.V = []
        self.ci = []


class DPPModel(object):
    def __init__(self, **kwargs):
        self.item_count = kwargs['item_count']
        self.item_embed_size = kwargs['item_embed_size']
        self.max_iter = kwargs['max_iter']
        self.epsilon = kwargs['epsilon']
        self.Z = {}
        self.load_last_item_num = 0
        self.last_ids = []

    def build_kernel_matrix(self):
        rank_score = np.random.random(size=(self.item_count)) / 10 ** 29  # 用户和每个item的相关性
        for i in range(5):
            rank_score[i] *= (3 * 10 ** 29)
        item_embedding = np.random.randn(self.item_count, self.item_embed_size)  # item的embedding
        item_embedding = item_embedding / np.linalg.norm(item_embedding, axis=1, keepdims=True)
        sim_matrix = np.dot(item_embedding, item_embedding.T)  # item之间的相似度矩阵
        self.kernel_matrix = rank_score.reshape((self.item_count, 1)) \
                             * sim_matrix * rank_score.reshape((1, self.item_count))
        assert math.isclose(self.kernel_matrix[0][0], rank_score[0] ** 2, rel_tol=1e-09, abs_tol=0.0)
        for i in range(self.item_count):
            if i < self.load_last_item_num:
                self.last_ids.append(i)
                rank_score[i] = 0.5
            item = Item(i, rank_score[i] * rank_score[i], item_embedding[i])
            self.Z[item.id] = item
        print(self.last_ids)

    def cal_sim(self, item, last_item):
        return np.dot(item.embedding, last_item.embedding)

    def cal_dpp_score(self, item, last_item):
        # print(item.id, last_item.id)
        Lij = item.r_score * last_item.r_score * self.cal_sim(item, last_item)
        ei = (Lij - np.dot(item.ci, last_item.ci)) / np.sqrt(last_item.d_score + 1e-9)
        item.ci.append(ei)
        item.d_score -= ei * ei
        if item.d_score < 0:
            print(item.id, item.d_score)
            item.d_score = 0

    def dpp(self):
        Yg = OrderedDict()
        last_item = None
        while len(Yg) < self.max_iter:
            for i, item in self.Z.items():
                if i in Yg.keys():
                    continue
                if last_item is not None:
                    self.cal_dpp_score(item, last_item)
                # print('i:', i, item.d_score)
            if last_item is not None:
                self.Z.pop(last_item.id)
            if len(Yg) >= self.load_last_item_num:
                if random.random() < 1.5:
                    j, last_item = max(self.Z.items(), key=lambda x: x[1].d_score)
                else:
                    j = list(self.Z.keys())[0]
                    last_item = self.Z[j]
            else:
                j, last_item = len(Yg), self.Z[len(Yg)]
            Yg[j] = last_item

        return [(k, v.d_score) for k, v in Yg.items()]

    def dpp2(self):
        Yg = OrderedDict()
        for i in self.last_ids:
            last_item = self.Z[i]
            for j in range(i + 1, self.item_count):
                item = self.Z[j]
                self.cal_dpp_score(item, last_item)

        # last_item = self.Z[self.last_ids[-1]]
        last_item = None
        for i in self.last_ids:
            self.Z.pop(i)
        while len(Yg) < self.max_iter:
            for i, item in self.Z.items():
                if i in Yg.keys():
                    continue
                if last_item is not None:
                    self.cal_dpp_score(item, last_item)
                # print('i:', i, item.d_score)
            if last_item is not None:
                self.Z.pop(last_item.id)
            j, last_item = max(self.Z.items(), key=lambda x: x[1].d_score)
            Yg[j] = last_item

        return [(k, v.d_score) for k, v in Yg.items()]

    def dpp3(self):
        c = np.zeros((self.max_iter, self.item_count))
        d = np.copy(np.diag(self.kernel_matrix))
        j = np.argmax(d)
        ans = []
        Yg = [j]
        ans.append((j, d[j]))
        iter = 0
        Z = list(range(self.item_count))
        while len(Yg) < self.max_iter:
            Z_Y = set(Z).difference(set(Yg))
            for i in Z_Y:
                ei = (self.kernel_matrix[i, j] - np.dot(c[:iter, j], c[:iter, i])) / np.sqrt(d[j])
                c[iter, i] = ei
                d[i] -= ei * ei
                # print('iter, i:', iter, i, d[i])
            d[j] = 0  # 把已经选过的置0
            j = np.argmax(d)
            Yg.append(j)
            ans.append((j, d[j]))
            iter += 1
        return ans


if __name__ == "__main__":
    kwargs = {
        'item_count': 100,
        'item_embed_size': 100,
        'max_iter': 10,
        'epsilon': 0.01
    }
    # np.random.seed(40)
    dpp_model = DPPModel(**kwargs)
    dpp_model.build_kernel_matrix()
    print(dpp_model.dpp())
    # print(dpp_model.dpp2())
    # print(dpp_model.dpp3())

    # import time
    # max_length = 1000
    # window_size = 10
    # t = time.time()
    # result_sw = dpp_model.dpp_sw(window_size)
    # print('sw algorithm running time: ' + '\t' + "{0:.4e}".format(time.time() - t))
