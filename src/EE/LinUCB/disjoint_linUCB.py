import numpy as np
import time

debug = True

class LinUCB:
    def __init__(self, articles=None):
        self.alpha = 0.25
        self.r1 = 0.6
        self.r0 = -16
        self.d = 6  # dimension of user features
        self.Aa = {}  # Aa : collection of matrix to compute disjoint part for each article a, d*d
        self.AaI = {}  # AaI : store the inverse of all Aa matrix

        self.ba = {}  # ba : collection of vectors to compute disjoin part, d*1
        self.theta = {}

        self.a_max = 0

        self.x = None
        self.xT = None

        if articles is not None:
            self.articles = articles
        else:
            self.articles = []

    def init_article_if_absent(self, article):
        if article not in self.articles:
            self.articles.append(article)
            self.Aa[article] = np.identity(self.d)  # 创建单位矩阵
            self.ba[article] = np.zeros((self.d, 1))
            self.AaI[article] = np.identity(self.d)
            self.theta[article] = np.zeros((self.d, 1))

    def init_all_articles(self, articles):
        for article in articles:
            self.init_article_if_absent(article)

    def update(self, reward):
        if reward == 1.0:
            r = self.r1
        elif reward == 0.0:
            r = self.r0
        else:
            print(f"reward:{reward} is not valid")
            r = 0

        self.Aa[self.a_max] += np.dot(self.x, self.xT)
        self.ba[self.a_max] += r * self.x
        self.AaI[self.a_max] = np.linalg.inv(self.Aa[self.a_max])
        self.theta[self.a_max] = np.dot(self.AaI[self.a_max], self.ba[self.a_max])

    def recommend(self, user_features, articles=None):
        xaT = np.array([user_features])
        xa = np.transpose(xaT)  # d * 1

        if articles is None:
            articles = self.articles

        AaI_tmp = np.array([self.AaI[article] for article in articles]) # (n, d, d)
        theta_tmp = np.array([self.theta[article] for article in articles]) #(n, d, 1)
        if debug:
            exploit = np.dot(xaT, theta_tmp)
            explore = np.sqrt(np.dot(np.dot(xaT, AaI_tmp), xa))
            idx = np.argmax(exploit + self.alpha * explore)
        else:
            idx = np.argmax(np.dot(xaT, theta_tmp) + self.alpha * np.sqrt(np.dot(np.dot(xaT, AaI_tmp), xa)))
        art_max = articles[idx]

        self.x = xa
        self.xT = xaT
        self.a_max = art_max

        return self.a_max

# def policy_evaluator(T: int, recommend, stream, articles):
#     total_r = 0.0
#     for i in range(T):
#         features, a, r = stream.next()
#         while recommend(features) != a:
#             features, a, r = stream.next()
#         total_r += r
#     return total_r/T

def get_train_data(file_name):
    with open(file_name, 'r', encoding='utf8') as f:
        for line in f:
            sp = line.strip().split('\u0001')
            features = list(map(lambda s: float(s), sp[0].split(',')))
            arm = sp[1]
            r = float(sp[2])
            yield (features, arm, r)


if __name__ == '__main__':
    linUCB = LinUCB()

    total_r = 0.0
    T = 1000

    # log内容，一行：
    # feature1 feature2 ...feature6\tarticle_id\tclick_or_not
    # 0.1 0.3 0 1 0 0\t123456\t0
    data_generator = get_train_data('/Users/jiananliu/work/AISEALs/data/ee/train.txt')
    # arms = []
    # for features, a, r in data_generator:
    #     if a not in arms:
    #         arms.append(a)
    #         # print(len(arms))
    # print(set(arms))

    for i in range(T):
        r = None
        num_not_use = 0
        t1 = time.time()
        for features, a, r in data_generator:
            num_not_use += 1
            linUCB.init_article_if_absent(a)
            if linUCB.recommend(features) == a:
                break
        t2 = time.time()
        if r is not None:
            linUCB.update(r)
            total_r += r

        if i % 1 == 0:
            print(f'evaluate step:{i} mean reward:{total_r/T} arm set:{len(linUCB.articles)}, num not use:{num_not_use}, use ts:{(t2-t1)*1000}ms, avg:{(t2-t1)*1000/num_not_use}ms')

