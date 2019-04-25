'''Q-learning demonstration on FrozenLake gym environment
'''
from os import system
from tqdm import tqdm
import numpy as np
import gym

# for training
MAX_EPISODES = 5000
MAX_MOVES_PER_EP = 200

# max number of hyperparamter combinations
MAX_COMBOS = 20
# max number of chances to be given to a hyperparameter combination
MAX_CHANCE = 5

# for testing
MAX_TESTS = 5000

# HYPERPARAM BOUNDS
HYP_BOUNDS = {
    'alpha': {'min': 0.65, 'max': 0.90},
    'gamma': {'min': 0.80, 'max': 0.95},
    'decay_factor': {'min': -1, 'max': 1},
}

# INITIAL HYPERPARAMS
init_hyperparams = {}
init_hyperparams['alpha'] = 0.77
init_hyperparams['gamma'] = 0.91
init_hyperparams['decay_factor'] = 0.44 # actual decay factor is 10 raised to this value

PERFORM_TUNING = False

def search_random_hyperparams(cur_hyperparams):
    '''Random Search implementation
    '''
    radius = 1.0 # radius of whole hypersphere
    search_rad = radius / 3 # radius of hypersphere to search for next hyperparams

    new_hyp = {}
    for hyp in cur_hyperparams:
        ave = (HYP_BOUNDS[hyp]['min'] + HYP_BOUNDS[hyp]['max']) / 2.0
        rng = HYP_BOUNDS[hyp]['max'] - HYP_BOUNDS[hyp]['min']

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

    return new_hyp

def tune_hyperparams(env):
    '''Hyperparameter tuning
    '''
    # use initial hyperparams
    hyperparams = init_hyperparams

    # for saving the best hyperparams
    best_hyperparams = hyperparams
    best_ave_wins = 0

    combo = 0
    while True:
        all_wins = []
        for i in tqdm(range(MAX_CHANCE)):
            Q, wins = get_q_table(env, hyperparams)
            all_wins.append(wins)

        # get average wins
        ave_wins = sum(all_wins) / len(all_wins)

        combo += 1
        print("[Combo %d]\nWin Summary\n  Average: %0.2f\n  Lowest: %0.2f\n  "
              "Highest: %0.2f" % (combo, ave_wins, min(all_wins), max(all_wins)))

        if ave_wins > best_ave_wins:
            best_hyperparams = hyperparams
            best_ave_wins = ave_wins
            print("[New Best]\n  alpha=%0.2f\n  gamma=%0.2f\n  "
                  "decay_factor=%0.2f\n"
                  % (best_hyperparams['alpha'],
                     best_hyperparams['gamma'],
                     10**best_hyperparams['decay_factor']))

        if combo == MAX_COMBOS:
            break

        hyperparams = search_random_hyperparams(best_hyperparams)

    return best_hyperparams

def get_q_table(env, hyperparams):
    '''Create and optimize q-table
    '''
    # get hyperparams
    alpha = hyperparams['alpha']
    gamma = hyperparams['gamma']
    df = hyperparams['decay_factor']

    # Q-table initalization
    # rows: states, cols: actions, cell value: expected reward
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    wins = 0
    for ep in range(MAX_EPISODES):
        cur_state = env.reset()

        for move in range(MAX_MOVES_PER_EP):
            #env.render()

            action = np.argmax(Q[cur_state, :] + \
                     np.random.randn(1, env.action_space.n)*(1./((ep+1)*(10**df))))

            prev_state = cur_state
            cur_state, reward, done, info = env.step(action)

            # update Q-table
            Q[prev_state, action] = Q[prev_state, action] + \
                                    alpha*(reward + gamma*(np.max(Q[cur_state, :])) - \
                                           Q[prev_state, action])

            # current episode ended
            if done:
                wins += reward
                break

    return Q, wins

def test_q_table(test_num, env, Q, random_agent=False):
    '''Run using optimized q-table or random agent
    '''
    wins = 0
    #Q = np.random.random([env.observation_space.n, env.action_space.n])
    for ep in range(MAX_TESTS):
        cur_state = env.reset()

        # fixed epsilon
        epsilon = 0.009
        if random_agent:
            epsilon = 1

        while True:
            #system('clear')
            #print("Test %d, Episode %d:" % (test_num+1, ep+1))
            #env.render()

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
                wins += reward
                break

    return wins

def run_tester(test_name, test_func, *args):
    '''Wrapper function for testing environment
    '''
    all_wins = []
    for i in tqdm(range(MAX_CHANCE)):
        wins = test_func(i, *args)
        all_wins.append(wins)

    # get average wins
    ave_wins = sum(all_wins) / len(all_wins)

    print("[%s]\nWin Summary\n  Average: %0.2f\n  Lowest: %0.2f\n  "
          "Highest: %0.2f\n  Ave. Win Rate: %0.4f\n"
          % (test_name, ave_wins, min(all_wins), max(all_wins), ave_wins/MAX_EPISODES))

def main():
    '''Main
    '''
    env = gym.make('FrozenLake8x8-v0')

    if PERFORM_TUNING:
        # perform Q-learning multiple times to determine best hyperparams
        print("Performing Q-learning for %d hyperparameter combos..." % MAX_COMBOS)
        hyperparams = tune_hyperparams(env)
    else:
        hyperparams = init_hyperparams

    # get the best Q-table
    best_wins = 0
    print("Getting best Q-table...")
    for i in tqdm(range(MAX_CHANCE)):
        Q, wins = get_q_table(env, hyperparams)
        print("Wins: %d" % wins)
        if wins >= best_wins:
            best_Q = Q

    #input("Random agent will be tested. Press Enter to continue...")
    print("Testing random agent...")
    # test how well a random agent performs
    run_tester("Random Agent", test_q_table,
               env, best_Q, True)

    #input("Learned Q-table will be tested. Press Enter to continue...")
    print("Testing learned Q-table...")
    # test how well a the learned agent performs
    run_tester("Agent Q", test_q_table,
               env, best_Q)

    env.close()

if __name__ == "__main__":
    main()
