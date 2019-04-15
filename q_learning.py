'''Q-learning demonstration on FrozenLake gym environment
'''
import math
import numpy as np
import gym

# for training
MAX_EPISODES = 5000
MAX_MOVES_PER_EP = 200

# max number of hyperparamter combinations
MAX_COMBOS = 1 #20
# max number of chances to be given to a hyperparameter combination
MAX_CHANCE = 5

# for testing
MAX_TRIES = 5

# HYPERPARAM BOUNDS
hyp_bounds = {
    'alpha': {'min': 0.65, 'max': 0.90},
    'gamma': {'min': 0.80, 'max': 0.95},
    'min_epsilon': {'min': 0.005, 'max': 0.03},
    'exp_decay_rate': {'min': 1, 'max': 5},
}

# - for Bellman equation
# alpha = 0.70    # learning rate
# gamma = 0.90    # discount rate (for rewards)
# min_epsilon = 0.01
# exp_decay_rate = 0.5/(10**2)

# add hyperparameter tuning

# random search:
#   sample from (min_alpha, max_alpha)
#   sample from (min_gamma, max_gamma)
#   sample from (min_epsilon, max_epsilon) ??
#   sample from (min_decay_rate, max_decay_rate)
#     - make sure these are normalized to (-1, 1)
#
#   perform q-learning
#   get wins? rewards?
#
#   OR
#
#   for max_tries:
#     perform q-learning
#
#   get average wins/rewards
#
#   if win/reward > prev:
#      best hyperparameters = (current hyperparameters)
#
#   sample_within_hypersphere()
#     - (if all params are normalized) imagine a unit sphere
#     - from current point (hyperparam coords) choose a value within specific radius
#     - if the next point will generate higher returns, then make it the next center
#
#   2 things to initialize here: the wins/rewards (=0), the hypersphere center (0,0?)

def search_random_hyperparams(cur_hyperparams):
    '''Random Search implementation
    '''
    radius = 1.0 # radius of whole hypersphere
    search_rad = radius / 2 # radius of hypersphere to search for next hyperparams

    new_hyp = {}
    for hyp in cur_hyperparams:
        ave = (hyp_bounds[hyp]['min'] + hyp_bounds[hyp]['max']) / 2.0
        rng = hyp_bounds[hyp]['max'] - hyp_bounds[hyp]['min']

        if hyp == 'exp_decay_rate':
            # for decay rate we perform random search on exponent of 10
            # this is used as multiplier / divider on the actual decay rate
            exponent = math.log10(0.5 / cur_hyperparams['exp_decay_rate'])
            scaled_hyp = ((exponent - ave) / rng) * (radius/0.5)

        else:
            # we add x2 to create hypersphere of radius 1
            scaled_hyp = ((cur_hyperparams[hyp] - ave) / rng) * (radius/0.5)

        # after scaling, choose uniformly for the next hyperparam
        low = scaled_hyp - search_rad
        high = scaled_hyp + search_rad
        # check if still within hypersphere
        low = low if (low >= -radius) else -radius
        high = high if (high <= radius) else radius 
        # sample, then de-normalize
        new_hyp[hyp] = np.random.uniform(low=low, high=high)
        new_hyp[hyp] = ((new_hyp[hyp] * (0.5/radius)) * rng) + ave
        if hyp == 'exp_decay_rate':
            new_hyp[hyp] = 0.5 / (10**new_hyp[hyp])

    return new_hyp

def tune_hyperparams(env):
    # use initial hyperparams
    hyperparams = {}
    hyperparams['alpha'] = 0.75
    hyperparams['gamma'] = 0.90
    hyperparams['min_epsilon'] = 0.01
    hyperparams['exp_decay_rate'] = 0.005

    # for saving the best hyperparams
    best_hyperparams = hyperparams
    best_ave_wins = 0

    combo = 0
    while True:
        all_wins = []
        for i in range(MAX_CHANCE):
            Q, wins = get_q_table(env, hyperparams)
            all_wins.append(wins)

        # get average wins
        ave_wins = sum(all_wins) / len(all_wins)
        if ave_wins > best_ave_wins:
            best_hyperparams = hyperparams
            best_ave_wins = ave_wins 

        combo += 1
        print("[Combo %d]\n  Average Wins: %0.2f\n  Lowest: %0.2f\n  "
              "Highest: %0.2f" % (combo, ave_wins, min(all_wins), max(all_wins)))
        if combo == MAX_COMBOS:
            break

        hyperparams = search_random_hyperparams(best_hyperparams)

    return hyperparams

def get_q_table(env, hyperparams):
    '''Create and optimize q-table
    '''
    # get hyperparams
    alpha = hyperparams['alpha']
    gamma = hyperparams['gamma']
    min_epsilon = hyperparams['min_epsilon']
    exp_decay_rate = hyperparams['exp_decay_rate']

    # Q-table initalization
    # rows: states, cols: actions, cell value: expected reward
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # initialize epsilon
    epsilon = 1.0

    wins = 0
    for ep in range(MAX_EPISODES):
        cur_state = env.reset()
        
        for move in range(MAX_MOVES_PER_EP):
            #env.render()

            # generate probability of exploitation
            exploit_prob = np.random.uniform()

            if exploit_prob > epsilon:
                # choose best action based on current state
                action = np.argmax(Q[cur_state, :]) # get index of best action
            else:
                action = env.action_space.sample()

            prev_state = cur_state
            cur_state, reward, done, info = env.step(action)

            # update Q-table
            Q[prev_state, action] = Q[prev_state, action] + \
                                    alpha*(reward + gamma*(np.max(Q[cur_state, :])) - \
                                           Q[prev_state,action])

            # current episode ended
            if done:
                wins += reward
                break

        # udpate epsilon
        epsilon = min_epsilon + ((1.0 - min_epsilon) * np.exp(-exp_decay_rate * ep))

    return Q, wins

def test_q_table(env, Q):
    '''Run using optimized q-table
    '''
    for ep in range(MAX_TRIES):
        cur_state = env.reset()
        
        # fix epsilon
        epsilon = 0.10

        while True:
            env.render()

            # generate probability of exploitation
            exploit_prob = np.random.uniform()

            if exploit_prob > epsilon:
                # choose best action based on current state
                action = np.argmax(Q[cur_state, :]) # get index of best action
            else:
                action = env.action_space.sample()

            cur_state, reward, done, info = env.step(action)

            # current episode ended
            if done:
                break

def main():
    '''Main
    '''
    env = gym.make('FrozenLake8x8-v0')
    
    # perform Q-learning multiple times to determine best hyperparams
    tune_hyperparams(env)

    #test_q_table(env, Q)
    
    env.close()

if __name__ == "__main__":
    main()