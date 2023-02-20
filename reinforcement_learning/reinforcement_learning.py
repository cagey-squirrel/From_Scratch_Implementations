from random import random, seed, choice
import os
from IPython.display import clear_output
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from math import log

def print_Q_values(state_action_Q_values):
    def stringify(float_num):
      str_value = str(float(float_num))[0:6]
      if len(str_value) < 6:
        str_value += '0'*(6-len(str_value))
      return str_value
    
    print(33 * ' ', end='')
    for key in state_action_Q_values:
      if key[0] == 'B':
        continue
      action_rewards = state_action_Q_values[key]
      val = stringify(action_rewards['up'])
      print(f"{val} {30 * ' '} ", end='')
    print()
    print()
    
    print(25 * ' ', end='')
    for key in state_action_Q_values:
      if key[0] == 'B':
        continue
      action_rewards = state_action_Q_values[key]
      val_left = stringify(action_rewards['left'])
      val_right = stringify(action_rewards['right'])
      print(f"{val_left} {3 * ' '}{key}{6 * ' '}{val_right}{13 * ' '} ", end='')
    print()
    print()
    
    
    print(33 * ' ', end='')
    for key in state_action_Q_values:
      if key[0] == 'B':
        continue
      action_rewards = state_action_Q_values[key]
      val = stringify(action_rewards['down'])
      print(f"{val} {30 * ' '} ", end='')
    print()
    print()
    print()
    print()
    
    print(32 * ' ', end='')
    for key in state_action_Q_values:
      if key[0] == 'A':
        continue
      action_rewards = state_action_Q_values[key]
      val = stringify(action_rewards['up'])
      print(f" {val} {68 * ' '}", end='')
    print()
    print()
    
    print(25 * ' ', end='')
    for key in state_action_Q_values:
      if key[0] == 'A':
        continue
      action_rewards = state_action_Q_values[key]
      val_left = stringify(action_rewards['left'])
      val_right = stringify(action_rewards['right'])
      print(f"{val_left} {3 * ' '}{key}{6 * ' '}{val_right}{51 * ' '} ", end='')
    print()
    print()
    
    
    print(32 * ' ', end='')
    for key in state_action_Q_values:
      if key[0] == 'A':
        continue
      action_rewards = state_action_Q_values[key]
      val = stringify(action_rewards['down'])
      print(f" {val} {68 * ' '}", end='')
    print()
    print()


def add_noise_to_action(action):
  random_value = random()
  
  if action == 'up':
    if random_value < 0.6:
      return 'up'
    elif random_value < 0.8:
      return 'left'
    else:
      return 'right'

  if action == 'down':
    if random_value < 0.6:
      return 'down'
    elif random_value < 0.8:
      return 'left'
    else:
      return 'right'

  if action == 'left':
    if random_value < 0.6:
      return 'left'
    elif random_value < 0.8:
      return 'up'
    else:
      return 'down'

  if action == 'right':
    if random_value < 0.6:
      return 'right'
    elif random_value < 0.8:
      return 'up'
    else:
      return 'down'
  
  raise Exception("Action must be one of up, down, left, right")


def make_a_move(current_state, action):
  '''
  Input:
    -current state
    -action to take
  Returns:
    -reward
    -new state
    -end==True
  '''
  if current_state in ['B1', 'B3', 'B5']:
    
    if current_state == 'B5':
      reward = 3
    else:
      reward = -1
    
    new_state = 'A1'
    is_end = True

    return reward, new_state, is_end

  noised_action = add_noise_to_action(action)
  reward = 0
  is_end = False

  if current_state == 'A1':
    if noised_action == 'left' or noised_action == 'up':
      new_state = current_state
    if noised_action == 'right':
      new_state = 'A2'
    if noised_action == 'down':
      new_state = 'B1'   
  
  elif current_state == 'A2':
    if noised_action == 'down' or noised_action == 'up':
      new_state = current_state
    if noised_action == 'right':
      new_state = 'A3'
    if noised_action == 'left':
      new_state = 'A1'
  
  elif current_state == 'A3':
    if noised_action == 'up':
      new_state = current_state
    if noised_action == 'right':
      new_state = 'A4'
    if noised_action == 'left':
      new_state = 'A2'
    if noised_action == 'down':
      new_state = 'B3'
  
  elif current_state == 'A4':
    if noised_action == 'down' or noised_action == 'up':
      new_state = current_state
    if noised_action == 'right':
      new_state = 'A5'
    if noised_action == 'left':
      new_state = 'A3'
  
  elif current_state == 'A5':
    if noised_action == 'right' or noised_action == 'up':
      new_state = current_state
    if noised_action == 'left':
      new_state = 'A4'
    if noised_action == 'down':
      new_state = 'B5'
  
  else:
    raise Exception(f"Invalid state {current_state}")
  
  return reward, new_state, is_end 


def q_learning(alpha, gamma, epsilon, num_iter):

  current_state = 'A1'

  #action_rewards = {'up':0, 'down':0, 'left':0, 'right':0}
  action_rewards = {'left':0, 'down':0, 'right':0, 'up':0}
  action_rewards_terminal1 = {'left':0, 'down':0, 'up':0, 'right':0}
  action_rewards_terminal2 = {'left':0, 'down':0, 'up':0, 'right':0}
  state_action_Q_values = {'A1':action_rewards.copy(), 'A2':action_rewards.copy(), 'A3':action_rewards.copy(), 'A4':action_rewards.copy(), 'A5':action_rewards.copy(), 'B1':action_rewards_terminal1.copy(), 'B3':action_rewards_terminal1.copy(), 'B5':action_rewards_terminal2.copy()}
  rewards = []
  v_values_per_state = defaultdict(lambda: [])

  for i in range(num_iter):
    is_over = False
    #alpha = log(i + 1) / (i + 1)
    while not is_over:
        random_value = random()

        if random_value < epsilon:
          action = choice(['up', 'down', 'left', 'right'])
        else:
          actions_Q_values = state_action_Q_values[current_state]
          action = max(actions_Q_values, key=actions_Q_values.get)

        reward, next_state, is_over = make_a_move(current_state, action)

        if not is_over:
          actions_Q_values = state_action_Q_values[next_state]
          max_reward_next_state = max(actions_Q_values.values())
          q = reward + gamma * max_reward_next_state
          state_action_Q_values[current_state][action] = (1-alpha) * state_action_Q_values[current_state][action] + alpha * q
        else:
          q = reward 
          state_action_Q_values[current_state][action] = (1-alpha) * state_action_Q_values[current_state][action] + alpha * q
        
        current_state = next_state 

    if (i+1) % 10 == 0:
      clear_output()
      print_Q_values(state_action_Q_values)
      optimal_policy = get_optimal_policy_from_Q(state_action_Q_values)
      reward = simulate_game(optimal_policy, 10)
      v_values = get_v_values((state_action_Q_values))
      for state_v in v_values:
        v_values_per_state[state_v].append(v_values[state_v])
      rewards.append(reward)
    
  return rewards, v_values_per_state


def get_v_values(state_action_Q_values):

  v_values_per_state = {}
  for state in state_action_Q_values:
    if state[0] == 'B':
      continue
    action_Q_values = state_action_Q_values[state]
    max_action_value = max(action_Q_values.values())
    v_values_per_state[state] = max_action_value

  return v_values_per_state


def get_optimal_policy_from_Q(state_action_Q_values):
  optimal_policy = state_action_Q_values.copy()

  for state in optimal_policy:
    actions_Q_values = state_action_Q_values[state]
    optimal_policy[state] = max(actions_Q_values, key=actions_Q_values.get)
  
  return optimal_policy


def simulate_game(policy, num_iter):

  policy_actions = list(policy.values())
  if policy_actions[0] == 'up' and policy_actions[2] == 'up' and policy_actions[4] == 'up':
    return 0, None

  if policy_actions[0] == 'up' and policy_actions[1] == 'left':
    return 0, None
  
  if policy_actions[0] == 'up' and policy_actions[2] == 'up' and policy_actions[3] == 'left':
    return 0, None

  rewards_sum = 0
  stuck_policy = None

  for i in range(num_iter):
    current_state = 'A1'
    is_end = False
    iteration = 0

    while not is_end:
      iteration += 1
      action = policy[current_state]
      noised_action = add_noise_to_action(action)
      reward, new_state, is_end = make_a_move(current_state, action)
      current_state = new_state

      if iteration == 100:
        stuck_policy = policy
        break

    
    rewards_sum += reward
  
  average_reward = rewards_sum / num_iter
  return average_reward


def main_q_learning():
  np.random.seed(3270)
  seed(3270)
  alpha = 0.01
  gamma = 0.99
  epsilon = 0.2
  num_iter = 20000
  num_iter_simulation = 10
  rewards, v_values_per_state = q_learning(alpha, gamma, epsilon, num_iter)
  plt.plot(range(len(rewards)), rewards)
  plt.show()

  print(v_values_per_state)
  for state in v_values_per_state:
    plt.plot(range(len(v_values_per_state[state])), v_values_per_state[state], label=state)
  
  plt.legend()
  plt.show()
  


main_q_learning()


# REINFORCE SOFTMAX S -> Q(S, A1), Q(S, A2), Q(S, A3), Q(S, A4)
def one_hot_features(state):
  state_row = state[0]
  state_col = int(state[1])

  features = np.zeros(8)

  if state_row == 'A':
    index = state_col - 1
  elif state_row == 'B':
    if state_col == 1:
      index = 5
    elif state_col == 3:
      index = 6
    elif state_col == 5:
      index = 7
  features[index] = 1

  return features[:, None]


def get_real_optimal_policy(theta):

  optimal_policy = {}

  states = ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B3', 'B5']
  for state in states:
    
    action, action_probs = get_real_best_softmax_action(theta, state)
    optimal_policy[state] = action
  
  return optimal_policy

def average_real_10_round_reward(theta):

  current_state = 'A1'
  total_reward = 0

  optimal_policy = get_real_optimal_policy(theta)
  
  # These policies cannot finish the game, they get stuck
  policy_actions = list(optimal_policy.values())
  if policy_actions[0] == 'up' and policy_actions[2] == 'up' and policy_actions[4] == 'up':
    return 0

  if policy_actions[0] == 'up' and policy_actions[1] == 'left':
    return 0
  
  if policy_actions[0] == 'up' and policy_actions[2] == 'up' and policy_actions[3] == 'left':
    return 0
  
  if policy_actions[1] == 'right' and policy_actions[2] == 'up' and policy_actions[3] == 'left':
    return 0

  for i in range(10):
    is_over = False
    current_state = 'A1'
    iterations = 0
    while not is_over:
        
      iterations += 1
      action = optimal_policy[current_state]

      reward, next_state, is_over = make_a_move(current_state, action)
      current_state = next_state 

      if iterations == 200:
          break

    
    total_reward += reward
  
  return total_reward / 10


def get_real_softmax_score(theta, state, action):

  actions = np.array(['up', 'down', 'left', 'right'])
  action_features = (actions == action).astype('float64')

  fi = one_hot_features(state)
  action_logits = np.dot(theta, fi)

  action_exp_logits = np.exp(action_logits)
  sum_action_exp_logits = np.sum(action_exp_logits)

  action_probs = (action_exp_logits / sum_action_exp_logits).flatten()

  diff = action_features - action_probs
    
  score = fi * diff.T

  return score.T

def get_real_best_softmax_action(theta, current_state):
  actions = ['up', 'down', 'left', 'right']

  action_probs_dict = {}

  fi = one_hot_features(current_state)
  action_logits = np.dot(theta, fi)
  action_exp_logits = np.exp(action_logits)
  sum_action_exp_logits = np.sum(action_exp_logits)

  action_probs = (action_exp_logits / sum_action_exp_logits).flatten()

  index = np.argmax(action_probs)

  for i, action in enumerate(actions):
    action_probs_dict[action] = action_probs[i]

  return actions[index], action_probs_dict



def reinforce_softmax(num_iter, alpha, gamma, print_iter):

  theta = np.random.rand(3, 8)
  theta_last_row = np.zeros((1, 8))

  theta = np.append(theta, theta_last_row, axis=0)  

  current_state = 'A1'
  rewards = []

  thetas1 = defaultdict(lambda: [])
  thetas2 = defaultdict(lambda: [])
  thetas3 = defaultdict(lambda: [])
  thetas = [thetas1, thetas2, thetas3]
  
  for i in range(num_iter):
    
    is_over = False
    episode = []
    reward = 0

    alpha = log(i+1) / (i + 1)

    treshold = 0.95

    if i < num_iter / 3:
      treshold = 0.95
    elif i < num_iter * 2 / 3:
      treshold = 0.5
    else:
      treshold = 0.1
    
    treshold = 1

    while not is_over:
        random_value = random()

        if random_value < treshold:
          action = choice(['up', 'down', 'left', 'right'])
        else:
          action, probs = get_real_best_softmax_action(theta, current_state)

        episode.append((current_state, action))

        reward, next_state, is_over = make_a_move(current_state, action)
        
        current_state = next_state 
    
    episode = reversed(episode)

    for state, action in episode:
      v = reward

      score = get_real_softmax_score(theta, state, action)
      theta = theta + alpha * score * v
      theta[3] = np.zeros((1,8))
      
      reward = reward * gamma
    
    if (i + 1) % print_iter == 0:

      for index_theta, theta_vector in enumerate(theta[0:3]):
        for index_inner_theta, theta_value in enumerate(theta_vector[0:5]):
          thetas[index_theta][index_inner_theta].append(theta_value)

      average_10_round_reward = average_real_10_round_reward(theta)
      rewards.append(average_10_round_reward)
    
  return theta, rewards, thetas



def main_real_reinforce_softmax():
  rseed = 3270
  np.random.seed(rseed)
  seed(rseed)

  num_iter = 20000
  alpha = 0.01
  gamma = 0.90
  print_iter = 10
  num_prints = num_iter // print_iter

  theta, rewards, thetas = reinforce_softmax(num_iter, alpha, gamma, print_iter)
  plt.plot(range(num_prints), rewards)
  plt.show()
  print(theta)

  outter_index_to_action_mapping = ['up', 'down', 'left', 'right']
  inner_index_to_state_mapping = ['A1', 'A2', 'A3', 'A4', 'A5']

  for outter_index, theta in enumerate(thetas):
    for inner_theta in theta:
      values = theta[inner_theta]
      plt.plot(range(len(values)), values, label=f'state {inner_index_to_state_mapping[inner_theta]}')
    
    plt.title(f"Action: {outter_index_to_action_mapping[outter_index]}")
    plt.legend()
    plt.show()
  

main_real_reinforce_softmax()