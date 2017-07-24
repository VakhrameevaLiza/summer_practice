import numpy as np
import gym

def running_average(x, window_size, mode='valid'):
    return np.convolve(x, np.ones(window_size) / window_size, mode=mode).max()

def check_solution(env, policy, n_episodes = 100, max_steps = 100, to_wrap = False, to_send = False, name2save = ''):
    ns = env.observation_space.n
    na = env.action_space.n
    count_dones = np.zeros(n_episodes)
    count_steps = np.zeros(n_episodes)
    if to_wrap:
        env = wrappers.Monitor(env, name2save)
        
    for i in range(n_episodes):
        observation = env.reset() 
        for step in range(max_steps): 
            action = policy[observation].argmax()
            observation, reward, done, info = env.step(action)
            count_dones[i] += reward
            count_steps[i] += 1
            if done:
                break
    return running_average(count_dones, 100).max()

    env.close()
    if to_wrap and to_send:
        gym.upload(name2save, api_key='sk_bExD4VfCSQukGlQkYKBhdQ')