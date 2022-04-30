import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
from reinforce import Agent


def train(start_epoch, scores, scores_window, gamma, n_episodes=1000):
    for i_episode in range(start_epoch + 1, n_episodes + 1):
        state = env.reset()
        agent.reset()
        step = 0
        while True:
            action, log_probs = agent.act(state.copy())
            state, reward, done, _ = env.step(action)
            step += 1
            reward = -100 if done and step < 450 else reward
            agent.step(reward, log_probs)
            if done:
                break
        agent.learn(gamma)
        scores_window.append(step)  # save most recent score
        scores.append(step)  # save most recent score

        print('Episodio {} -> Mejor Puntaje: {}'.format(i_episode, np.mean(scores_window)))
    return scores


def play():
    state = env.reset()
    agent.reset()
    score = 0
    while True:
        action, _ = agent.act(state)
        env.render()
        next_state, reward, done, _ = env.step(action)
        score += reward
        state = next_state
        if done:
            print("Puntaje: ", score)
            break
    env.close()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    env = gym.make('CartPole-v1')

    GAMMA = 1.0
    LR = 1e-2

    agent = Agent(state_size=4, action_size=env.action_space.n, learning_rate=LR, device=device)

    epoch = 0
    scores = []
    window = deque(maxlen=100)

    scores = train(epoch, scores, window, GAMMA, 4000)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    play()
