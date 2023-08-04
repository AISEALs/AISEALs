import sys
import os
from os.path import dirname, basename, exists
import numpy as np
from src_learner import ESLearner, CMAESLearner, SCMLearner
from src_utils import normalize


def get_learner(theta_dim):
    learner = SCMLearner(theta_dim, "SGD", 0.01, 
                        mu=1.0, 
                        weight_type="raw",
                        use_C=0,
                        cc=0.0,
                        ccov=0.0)

    return learner


def get_init_weight(theta_dim, home_path):
    res_thetas = []

    with open(home_path + "/data/one_step_tmp/theta.txt", 'r') as rf:
        for line in rf:
            thetas_list = line.strip().split("\t")
            if len(thetas_list) == theta_dim:
                res_thetas = [float(k) for k in thetas_list]
    
    return np.asarray(res_thetas)
    

def merge_gradients(list_g, gamma):
    """
    g_0 + gamma * g_1 + gamma^2 * g_2 + ...
    and then calculate weighted sum
    """
    num_g = len(list_g)
    gammas = np.reshape(np.cumprod([1] + [gamma] * (num_g - 1)), [-1, 1])
    res_g = np.sum(np.array(list_g) * gammas, 0) / np.sum(gammas)
    return res_g


def es_alg(replay_memory, home_path, lr):
    theta_dim = replay_memory[0]['noises'].shape[1]
    learner = get_learner(theta_dim)
    step = learner.optimizer.t
    # lr_reduce_step = 2
    lr_reduce_step = float(lr) # 2021-12-23 17:41
    with open("lr.txt", "w") as fw:
    	fw.write(str(lr_reduce_step) + "\n")
    sigma = 0.3

    learner.optimizer.stepsize = (lr_reduce_step ** int(step))
    if learner.optimizer.stepsize < 0.0001:
        learner.optimizer.stepsize = 0.0001
    
    anchor_theta = get_init_weight(theta_dim, home_path)

    #smfw = "1"
    #if learner.optimizer.t == 0:
    #    anchor_theta = get_init_weight(smfw)
    #else:
    #    anchor_theta = learner.get_weight()

    learner.set_weight(anchor_theta)
    list_g = []
    # sum_rewards = 0.0
    # len_rewards = 0

    for i, replay_dict in enumerate(replay_memory):
        a = replay_dict['noises']
        b = replay_dict['thetas']
        c = np.reshape(anchor_theta, [1, -1])
        print(anchor_theta.shape)
        print(a.shape, b.shape, c.shape)
        #模型纠偏
        replay_noises = replay_dict['noises'] + replay_dict['thetas'] - c
        #没出现的特征 不进行计算
        replay_noises[np.where(replay_dict['noises'] == 0)] = 0
        #replay_noises = replay_dict['noises']
        replay_rewards = replay_dict['rewards']
        replay_playTime = replay_dict['playTime']
        # sum_rewards += np.sum(replay_rewards)
        # len_rewards += replay_rewards.shape[0]

        #replay_rewards = normalize(replay_rewards, a_min=0.0, a_max=10000.0) 
        g = learner.calculate_gradient_new(replay_noises, replay_rewards, sigma)
        
        list_g.append(g)
    merge_g = merge_gradients(list_g, 0.75)
    #norm_g, update_ratio = learner.update(merge_g)
    norm_g = learner.update_new(merge_g)
    new_theta = learner.get_weight()
    
    return new_theta


def read_real_data(file):
    replay_dict = {'thetas': [], 'noises': [], 'rewards': [], 'playTime': []}
    with open(file, 'r') as f:
        for line_id, line in enumerate(f):
            line_content = line.strip().split('\t')
            if len(line_content) == 4:
                key, theta_list, noise_list, rewards = line_content
            else:
                continue


            reward_pt = rewards.split("@")
            #if len(reward_pt) != 2:
                #continue
            reward = float(reward_pt[0])
            playTime = 1.0

            noise_segs = [float(x) for x in noise_list.strip().split(',')]
            theta_segs = [float(x) for x in theta_list.strip().split(',')]

            noise = np.array(noise_segs)
            theta = np.array(theta_segs)
            
            replay_dict['thetas'].append(theta)
            replay_dict['playTime'].append(playTime)
            replay_dict['noises'].append(noise)
            replay_dict['rewards'].append(reward)
    print("read_real_data from:"+ file)
    for name in replay_dict:
        replay_dict[name] = np.array(replay_dict[name])
        print(name,len(replay_dict[name]))

    return replay_dict


def read_list_real_data(home_path):
    list_file = [home_path + "/data/one_step_tmp/allsample.txt"]
    replay_memory = [read_real_data(file) for file in list_file]
    return replay_memory


def dump_theta(theta, home_path):
    file = home_path + "/data/one_step_tmp/theta.txt"

    if not exists(dirname(file)) and dirname(file) != '':
        os.makedirs(dirname(file))
    with open(file, 'w') as f:
        str_theta = '\t'.join([str(x) for x in theta])
        f.write(str_theta)    
        
        
def write_to_file(lines, outputfile):
    outfile = open(outputfile, 'a')
    for line in lines:
        outfile.write(line + '\n')
    outfile.close()        
    
    
if __name__ == '__main__':
    lr = sys.argv[1]
    home_path = '.'
    print("read_list_real_data")
    curr_replay_dict = read_list_real_data(home_path)
    print("in es_alg")
    new_theta = es_alg(curr_replay_dict, home_path, lr)
    print("out es_alg")
    dump_theta(new_theta, home_path)

    record_theta_file = home_path + "/data/theta_record.txt"
    str_theta = '\t'.join([str(x) for x in new_theta])
    write_to_file([str_theta], record_theta_file)
