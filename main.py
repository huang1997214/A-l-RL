from RL.model import RL_model
from ex1_jobEnv import SchedulingEnv
import torch
import numpy as np

mod = RL_model(11,10)
env = SchedulingEnv(15)
job_c = 1
global_step = 0
start_learn = 100
target_update_freq = 300

epoch = 10

for ep in range(epoch):
    global_step = 0
    learn_step = 0
    job_c = 1
    env.reset(15)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    while True:
        global_step += 1
        finish, job_attrs = env.workload(job_c)
        DQN_state = env.getState(job_attrs, 4)
        #print(DQN_state)
        if global_step > start_learn or ep>0:
            p = np.random.randint(10)
            if p < 8:
                act = mod(torch.FloatTensor(DQN_state))
                act = np.argmax(act.detach().numpy())
            else:
                act = action = np.random.randint(10)
        else:
            act = action = np.random.randint(10)
        #print(act)
        reward_DQN = env.feedback(job_attrs, act, 4)
        #print(reward_DQN)
        if global_step!=1:
            mod.store_buffer(last_state, last_action, last_reward, DQN_state)
        if global_step > start_learn:
            mod.learn()
            learn_step += 1
        if learn_step % target_update_freq == 0:
            mod.update_target()
        last_state = DQN_state
        last_action = act
        last_reward = reward_DQN
        if finish:
            break
        else:
            job_c += 1

