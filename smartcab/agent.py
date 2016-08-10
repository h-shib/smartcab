import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.alpha = 0.6   # learning rate
        self.gamma = 0.9   # discount factor
        self.epsilon = 0   # epsilon greedy factor
        self.Q_values = {} # key: state, value: {action: value}
        self.trial_count = 0
        self.rewards = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def get_action(self, state, epsilon):
        if state not in self.Q_values:
            self.Q_values[state] = {None: 10, 'forward': 10, 'left': 10, 'right': 10}
        #if random.random() < epsilon:
            return random.choice(self.env.valid_actions)
        else:
            return max(self.Q_values[state].items(), key=lambda x:x[1])[0]

    def get_max_Q_value(self, state):
        return max(self.Q_values[state].values()) if state in self.Q_values else 10

    def learn_policy(self, state, action, reward, next_state):
        self.Q_values[state][action] = (1 - self.alpha) * self.get_max_Q_value(state) + self.alpha * (reward + self.gamma * self.get_max_Q_value(next_state))


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.trial_count += 1

        # TODO: Update state
        #self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'])
        bool_for_left = (inputs['light'] == 'green') and (inputs['oncoming'] == None or inputs['oncoming'] == 'left')
        bool_for_right = (inputs['light'] == 'green') or (inputs['left'] != 'forward')
        self.state = (self.next_waypoint, inputs['light'], bool_for_left, bool_for_right)

        # TODO: Select action according to your policy
        self.epsilon = 1 / float(self.trial_count + 1)
        action = self.get_action(self.state, self.epsilon)

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.rewards += reward

        # TODO: Learn policy based on state, action, reward
        next_waypoint = self.planner.next_waypoint()
        next_inputs = self.env.sense(self)

        next_bool_for_left = (inputs['light'] == 'green') and (inputs['oncoming'] == None or inputs['oncoming'] == 'left')
        next_bool_for_right = (inputs['light'] == 'green') or (inputs['left'] != 'forward')
        next_state = (next_waypoint, next_inputs['light'], next_bool_for_left, next_bool_for_right)
        self.learn_policy(self.state, action, reward, next_state)

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
