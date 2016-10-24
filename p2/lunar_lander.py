from sklearn.neural_network import MLPRegressor
import numpy as np
import gym
import argparse


class LunarLander():
    def __init__(self,
                 k_actions,
                 alpha=1e-5,
                 gamma=0.9,
                 start_epsilon=1.,
                 min_epsilon=0.1,
                 epsilon_decay=0.99,
                 consecutive=100,
                 minibatch_size=40,
                 training_steps=5,
                 max_replay_memory=20000,
                 average_reward=200.,
                 output_file='output'):
        self._init = True
        self._K_ACTIONS = k_actions
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = start_epsilon
        # create four neural networks
        # TODO: optimize this to use multi-labels e.g. X=[obs], y=[[0, 0, reward, 0]]
        self._nn = [MLPRegressor(warm_start=True) for _ in xrange(self._K_ACTIONS)]
        self._steps = 0
        self._total_reward = 0.
        self._best_reward = -999.
        self._average_reward = average_reward
        self._D = [[] for _ in xrange(self._K_ACTIONS)]
        self._consecutive_actions = [consecutive for _ in xrange(self._K_ACTIONS)]
        self._state = None
        self._action = None
        self._rewards = []
        self._trained = np.zeros(self._K_ACTIONS)
        self._MINIBATCH_SIZE = minibatch_size
        self._TRAINING_STEPS = training_steps
        self._MAX_REPLAY_MEMORY = max_replay_memory
        self._EPSILON_DECAY = epsilon_decay
        self._MIN_EPSILON = min_epsilon
        self.TRAINED_EPISODES = 100

        self._STATE_INDEX = 0
        self._ACTION_INDEX = 1
        self._REWARD_INDEX = 2
        self._NEXT_STATE_INDEX = 3
        self._TERMINAL_INDEX = 4

        filename = output_file + '-{_gamma}-{_EPSILON_DECAY}.csv'.format(**self.__dict__)
        self.output = open(filename, 'w')
        line = ','.join([str(x) for x in ['episode', 'reward', 'average_reward']])
        self.output.write(line + '\n')

    def _print_average_reward(self, average_reward):
        print ('Episode', episode,
               'Reward', self._total_reward,
               'Average Reward', round(average_reward, 2),
               'Best Reward', self._best_reward)

    def _decay_epsilon(self):
        self._epsilon = max(self._epsilon * self._EPSILON_DECAY, self._MIN_EPSILON)

    def _store_transition(self, state, action, reward, next_state, terminal):
        transition = [
            np.copy(state),
            action,
            reward,
            np.copy(next_state),
            terminal
        ]

        self._D[action].append(transition)

    def _write_to_file(self, average_reward):
        # line = "\t".join([str(x) for x in [i, episode_steps, episode_reward, reward, epsilon, average]])
        line = ','.join([str(x) for x in [episode, self._total_reward, average_reward]])
        self.output.write(line + '\n')

    def predict(self, observation, reward, terminal, episode):
        # save total reward for current episode
        self._total_reward += reward

        # on terminal, reset total reward to 0 and decay the epsilon
        if terminal:
            if self._total_reward > self._best_reward:
                self._best_reward = self._total_reward
            # save total reward to the rewards array to calculate average reward
            self._rewards.append(self._total_reward)
            # calculate average
            average_reward = np.mean(self._rewards[-self.TRAINED_EPISODES:])
            # print
            self._print_average_reward(average_reward)
            # save to file
            self._write_to_file(average_reward)
            # reset total reward to 0 for new episode
            self._total_reward = 0.
            # decay epsilon
            self._decay_epsilon()
            if average_reward >= self._average_reward:
                self.output.close()
                return -1

        next_state = observation.reshape((1, len(observation)))
        self._steps += 1

        # initially choose random action
        if self._state is None:
            self._state = np.copy(next_state)
            action = np.random.randint(self._K_ACTIONS)
            self._action = action
            return action

        # store the transition <s, a, r, s'> to D
        self._store_transition(self._state, self._action, reward, next_state, terminal)
        self._state = np.copy(next_state)

        # if the memory is full, remove the oldest stored transition
        for i in xrange(self._K_ACTIONS):
            while len(self._D[i]) > self._MAX_REPLAY_MEMORY:
                self._D[i].pop(0)

        if self._steps % self._TRAINING_STEPS == 0:
            self._fit()

        if self._consecutive_actions[self._action] > 0:
            self._consecutive_actions[self._action] -= 1
            action = self._action
        elif np.random.random() >= self._epsilon:
            # with probability epsilon, select action randomly otherwise use the prediction
            values = [self._nn[i].predict(self._state) if self._trained[i] else 0. for i in xrange(self._K_ACTIONS)]
            action = np.argmax(values)
        else:
            action = np.random.randint(0, self._K_ACTIONS)

        self._action = action

        return action

    def _fit(self):
        for action in xrange(self._K_ACTIONS):
            if self._MINIBATCH_SIZE > len(self._D[action]):
                continue

            indices = np.random.permutation(len(self._D[action]))[:self._MINIBATCH_SIZE]
            selected_transitions = [self._D[action][i] for i in indices]

            states = np.concatenate([t[self._STATE_INDEX] for t in selected_transitions])
            rewards = np.array([t[self._REWARD_INDEX] for t in selected_transitions])
            next_states = np.concatenate([t[self._NEXT_STATE_INDEX] for t in selected_transitions])
            done = np.array([t[self._TERMINAL_INDEX] for t in selected_transitions])

            future_rewards = np.array([self._nn[i].predict(next_states).reshape(self._MINIBATCH_SIZE,) if self._trained[i] \
                else np.zeros(self._MINIBATCH_SIZE) for i in xrange(self._K_ACTIONS)]).T

            values = np.copy(rewards)
            values += (1. - done) * self._gamma * np.max(future_rewards, axis=1)

            self._nn[action].partial_fit(states, values)
            self._trained[action] = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Q-Network Lunar Lander Agent')
    parser.add_argument('-n', '--name', type=str, default='LunarLander-v2', help='Game name')
    parser.add_argument('-e', '--episodes', type=int, default=10000, help='Number of episodes')
    parser.add_argument('-es', '--epsilon-start', type=float, default=1., help='Init epsilon')
    parser.add_argument('-ed', '--epsilon-decay', type=float, default=.99, help='Epsilon decay rate')
    parser.add_argument('-em', '--epsilon-min', type=float, default=.0, help='Min epsilon')
    parser.add_argument('-g', '--gamma', type=float, default=.995)
    parser.add_argument('-c', '--consecutive', type=int, default=1, help='Number of times to repeat random action')
    parser.add_argument('-bs', '--batch-size', type=int, default=40, help='Size of mini-batch')
    parser.add_argument('-br', '--batch-rate', type=int, default=5, help='Rate to train the mini-batch')
    parser.add_argument('-D', '--replay-memory', type=int, default=20000, help='Size of replay memory')
    parser.add_argument('-avg', '--average-reward', type=float, default=200.,
                        help='Terminate when average reward is reached')
    parser.add_argument('-o', '--output', type=str, default='output', help='Output file prefix')

    args = parser.parse_args()

    env = gym.make(args.name)
    env.monitor.start(args.output, force=True, video_callable=False)

    # TODO: get parameters from args
    learner = LunarLander(env.action_space.n,
                          alpha=0.01,
                          gamma=args.gamma,
                          start_epsilon=args.epsilon_start,
                          min_epsilon=args.epsilon_min,
                          epsilon_decay=args.epsilon_decay,
                          consecutive=args.consecutive,
                          minibatch_size=args.batch_size,
                          training_steps=args.batch_rate,
                          max_replay_memory=args.replay_memory,
                          output_file=args.output)

    # training step
    for episode in xrange(1, args.episodes + 1):
        observation = env.reset()
        reward = 0
        done = False
        action = learner.predict(observation, reward, done, episode)
        t = 0

        while not done and t < 1000 and action != -1:
            t += 1
            # env.render()
            observation, reward, done, info = env.step(action)
            action = learner.predict(observation, reward, done, episode)

        if action == -1:
            break

    # test for 100 trials
    d = {'output_file': args.output, 'gamma': args.gamma, 'epsilon_decay': args.epsilon_decay}
    filename = '{output_file}-test-{gamma}-{epsilon_decay}.csv'.format(**d)
    learner.output = open(filename, 'w')
    for episode in xrange(1, 101):
        observation = env.reset()
        reward = 0
        done = False
        action = learner.predict(observation, reward, done, episode)
        t = 0

        while not done and t < 1000 and action != -1:
            t += 1
            # env.render()
            observation, reward, done, info = env.step(action)
            action = learner.predict(observation, reward, done, episode)

        if action == -1:
            break

    env.monitor.close()
    env.close()
