import numpy as np
import math


class DPPModel(object):
    def __init__(self, **kwargs):
        self.item_count = kwargs['item_count']
        self.item_embed_size = kwargs['item_embed_size']
        self.max_iter = kwargs['max_iter']
        self.epsilon = kwargs['epsilon']

    def build_kernel_matrix(self):
        rank_score = np.random.random(size=(self.item_count))  # 用户和每个item的相关性
        item_embedding = np.random.randn(self.item_count, self.item_embed_size)  # item的embedding
        item_embedding = item_embedding / np.linalg.norm(item_embedding, axis=1, keepdims=True)
        sim_matrix = np.dot(item_embedding, item_embedding.T)  # item之间的相似度矩阵
        self.kernel_matrix = rank_score.reshape((self.item_count, 1)) \
                             * sim_matrix * rank_score.reshape((1, self.item_count))
        assert math.isclose(self.kernel_matrix[0][0], rank_score[0] ** 2, rel_tol=1e-09, abs_tol=0.0)

    def dpp(self):
        c = np.zeros((self.max_iter, self.item_count))
        d = np.copy(np.diag(self.kernel_matrix))
        j = np.argmax(d)
        Yg = [j]
        iter = 0
        Z = list(range(self.item_count))
        while len(Yg) < self.max_iter:
            Z_Y = set(Z).difference(set(Yg))
            for i in Z_Y:
                # when iter == 0: c[:iter, j] is empty
                ei = (self.kernel_matrix[j, i] - np.dot(c[:iter, j], c[:iter, i])) / np.sqrt(d[j])
                c[iter, i] = ei
                d[i] = d[i] - ei * ei
                print('iter, i:', iter, i, d[i])
            d[j] = 0 # 把已经选过的置0
            j = np.argmax(d)
            if d[j] < self.epsilon:
                break
            Yg.append(j)
            iter += 1

        return Yg

    def dpp_sliding_window(self, window_size):
        c = np.zeros((self.max_iter, self.item_count))
        d = np.copy(np.diag(self.kernel_matrix))
        j = np.argmax(d)
        Yg = [j]
        iter = 0
        Z = list(range(self.item_count))
        while len(Yg) < self.max_iter:
            Z_Y = set(Z).difference(set(Yg))
            for i in Z_Y:
                # when iter == 0: c[:iter, j] is empty
                ei = (self.kernel_matrix[j, i] - np.dot(c[:iter, j], c[:iter, i])) / np.sqrt(d[j])
                c[iter, i] = ei
                d[i] = d[i] - ei * ei
                # print('iter, i:', iter, i, d[i])
            if len(Yg) >= window_size:
                pass
            d[j] = -np.inf  # 把已经选过的置-inf
            j = np.argmax(d)
            if d[j] < self.epsilon:
                break
            Yg.append(j)
            iter += 1
        return Yg

    def dpp_sw(self, window_size, epsilon=1E-10):
        kernel_matrix = self.kernel_matrix
        max_length = self.max_iter
        item_size = kernel_matrix.shape[0]
        V = np.zeros((max_length, max_length))
        cis = np.zeros((max_length, item_size))
        d = np.copy(np.diag(kernel_matrix))
        selected_item = np.argmax(d)
        Yg = [selected_item]
        window_left_index = 0
        while len(Yg) < max_length:
            k = len(Yg) - 1
            ci_optimal = cis[window_left_index:k, selected_item]
            di_optimal = math.sqrt(d[selected_item])
            V[k, window_left_index:k] = ci_optimal
            V[k, k] = di_optimal
            elements = kernel_matrix[selected_item, :]
            eis = (elements - np.dot(ci_optimal, cis[window_left_index:k, :])) / di_optimal
            cis[k, :] = eis
            d -= np.square(eis)
            if len(Yg) >= window_size:
                window_left_index += 1
                for ind in range(window_left_index, window_size):
                    t = math.sqrt(V[ind, ind] ** 2 + V[ind, window_left_index - 1] ** 2)
                    c = t / V[ind, ind]
                    s = V[ind, window_left_index - 1] / V[ind, ind]
                    V[ind, ind] = t
                    V[ind + 1:window_size, ind] += s * V[ind + 1:window_size, window_left_index - 1]
                    V[ind + 1:window_size, ind] /= c
                    V[ind + 1:window_size, window_left_index - 1] *= c
                    V[ind + 1:window_size, window_left_index - 1] -= s * V[ind + 1:k + 1, ind]
                    cis[ind, :] += s * cis[window_left_index - 1, :]
                    cis[ind, :] /= c
                    cis[window_left_index - 1, :] *= c
                    cis[window_left_index - 1, :] -= s * cis[ind, :]
                d += np.square(cis[window_left_index - 1, :])
            d[selected_item] = -np.inf
            selected_item = np.argmax(d)
            if d[selected_item] < epsilon:
                break
            Yg.append(selected_item)
        return Yg

if __name__ == "__main__":
    kwargs = {
        'item_count': 100,
        'item_embed_size': 100,
        'max_iter': 10,
        'epsilon': 0.01
    }
    dpp_model = DPPModel(**kwargs)
    dpp_model.build_kernel_matrix()
    print(dpp_model.dpp())

    # import time
    # max_length = 1000
    # window_size = 10
    # t = time.time()
    # result_sw = dpp_model.dpp_sw(window_size)
    # print('sw algorithm running time: ' + '\t' + "{0:.4e}".format(time.time() - t))
