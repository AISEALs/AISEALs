import numpy as np
import random
from agent.timer import timer
from agent.env.Snake import Snake
from agent.env.GridWorld import GridWorld


class TableAgent:
    def __init__(self, env, reward_table):
        self.env = env
        self.reward = reward_table
        self.state_num = env.state_num
        self.act_num = env.action_num
        self.policy = np.ones((self.state_num, self.act_num), dtype=np.int)
        self.value_pi = np.zeros(self.state_num)
        self.value_q = np.zeros((self.state_num, self.act_num))
        self.gamma = 1

    def policy_evaluation(self):
        k = 1
        new_value_pi = np.zeros_like(self.value_pi)  # init pi
        while True:
            for i in range(1, self.state_num - 1):
                # new_value_pi[i] = np.dot(self.trans_table[self.policy[i], i, :], self.reward + self.gamma * new_value_pi)
                p = self.env.state_transition_table(i, self.policy[i])
                new_value_pi[i] = np.dot(p, self.reward + self.gamma * self.value_pi)

            print(str(k) + ':' + str(new_value_pi))

            k += 1
            diff = np.sqrt(np.sum(np.power(self.value_pi - new_value_pi, 2)))
            self.value_pi = new_value_pi.copy()
            if diff < 1e-6:
                break
        print(k)

    def policy_improvement(self):
        new_policy = np.zeros_like(self.policy)
        for i in range(1, self.state_num - 1):
            for j in range(self.act_num):
                policy = np.zeros(self.act_num, np.int)
                policy[j] = 1
                p = self.env.state_transition_table(i, policy)
                self.value_q[i, j] = self.reward[i] + self.gamma * np.dot(p, self.value_pi)
            # update policy
            # max_act = np.argmax(self.value_q[i, :])
            # new_policy[i][max_act] = 1
            winner = np.argwhere(self.value_q[i] >= np.amax(self.value_q[i]) - 1e-6)
            for action in winner.flatten().tolist():
                new_policy[i][action] = 1

        # (101, ) =  np.argmax((2, 101, 101) * (101,), axis=0)
        # new_policy = np.argmax(np.dot(self.trans_table, self.reward + self.gamma * self.value_pi), axis=0)
        # is_same = np.all(np.equal(new_policy, new_policy2))
        # print("is_same:" + str(is_same))
        print("old_policy:" + str(self.policy))
        print("new_policy:" + str(new_policy))
        if np.all(np.equal(new_policy, self.policy)):
            return False
        else:
            self.policy = new_policy.copy()
            return True

    def policy_iteration(self):
        iteration = 0
        while True:
            iteration += 1
            with timer('Timer PolicyEval'):
                self.policy_evaluation()
                # self.policy_evaluation3()
            with timer('Timer PolicyImpove'):
                ret = self.policy_improvement()
            if not ret:
                print('Iter {} rounds converge'.format(iteration))
                print(self.value_pi)
                print(self.value_q[self.policy[0]])
                break


def policy_iteration_demo():
    # env = Snake(10, [3, 6])
    env = GridWorld(4, 4)
    agent = TableAgent(env, env.reward_table())
    agent.policy_iteration()
    print("final policy:" + str(agent.policy))
    return agent.policy


def value_iteration():
    env = Snake(10, [3, 6])
    table = env.state_transition_table()
    reward_table = env.reward_table()
    v = np.zeros(101)
    v_new = np.zeros_like(v)

    alpha = 0.8
    total_count = 0
    while True:
        # synchronous
        v_matrix = np.dot(table, v)   # shape=(2,101)
        for i in range(1, 101):
            v_new[i] = reward_table[i] + alpha * max(v_matrix[:, i])
        #asynchronous
        v_new = v.copy()
        for i in range(1, 101):
            v_new[i] = reward_table[i] + alpha * max(np.dot(table, v_new)[:, i])

        total_count += 1
        diff = np.sum(np.power(v - v_new, 2))
        if diff < 1e-6:
            break
        else:
            v = v_new.copy()
    pi = np.zeros(101, dtype=np.int32)
    for i in range(1, 101):
        pi[i] = np.argmax(np.dot(table[:, i, :], v_new))
    print(pi, v_new, total_count)
    return pi


def q_learning_demo():
    env = Snake(10, [3, 6])
    table = env.state_transition_table()
    reward_table = env.reward_table()
    q_matrix = np.zeros((100, 2))

    alpha = 0.8
    while True:
        start_state = 1
        cur_state = start_state
        while cur_state != 100:
            action = random.randint(0, 1)


if __name__ == "__main__":
    # q_learning_demo()
    # vi_rt = value_iteration()
    pi_rt = policy_iteration_demo()

    # print np.array_equal(vi_rt, pi_rt)
    # print np.equal(vi_rt, pi_rt)
