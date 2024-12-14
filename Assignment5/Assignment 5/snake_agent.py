import numpy as np
import helper
import random

#   This class has all the functions and variables necessary to implement snake game
#   We will be using Q learning to do this

class SnakeAgent:

    #   This is the constructor for the SnakeAgent class
    #   It initializes the actions that can be made,
    #   Ne which is a parameter helpful to perform exploration before deciding next action,
    #   LPC which ia parameter helpful in calculating learning rate (lr) 
    #   gamma which is another parameter helpful in calculating next move, in other words  
    #            gamma is used to blalance immediate and future reward
    #   Q is the q-table used in Q-learning
    #   N is the next state used to explore possible moves and decide the best one before updating
    #           the q-table
    def __init__(self, actions, Ne, LPC, gamma):
        self.actions = actions
        self.Ne = Ne
        self.LPC = LPC
        self.gamma = gamma
        self.reset()
 
        self.s = None
        self.a = None
        self.points = 0
        self.epsilon = .1
        self.alpha = .7

        # Create the Q and N Table to work with
        self.Q = helper.initialize_q_as_zeros()
        self.N = helper.initialize_q_as_zeros()


    #   This function sets if the program is in training mode or testing mode.
    def set_train(self):
        self._train = 1

     #   This function sets if the program is in training mode or testing mode.       
    def set_eval(self):
        self._train = 0

    #   Calls the helper function to save the q-table after training
    def save_model(self):
        helper.save(self.Q)

    #   Calls the helper function to load the q-table when testing
    def load_model(self):
        self.Q = helper.load()

    #   resets the game state
    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    #   This is a function you should write. 
    #   Function Helper:IT gets the current state, and based on the 
    #   current snake head location, body and food location,
    #   determines which move(s) it can make by also using the 
    #   board variables to see if its near a wall or if  the
    #   moves it can make lead it into the snake body and so on. 
    #   This can return a list of variables that help you keep track of
    #   conditions mentioned above.
    def helper_func(self, state):
        # print("In helper_func")
        
        idx = [0] * 8
        snake_head_x, snake_head_y = state[0], state[1]
        snake_body_x = [body[0] for body in state[2]]
        snake_body_y = [body[1] for body in state[2]]
        food_x, food_y = state[3], state[4]
        
        ADJ_WALL_X = 0
        ADJ_WALL_Y = 1
        FOOD_DIR_X = 2
        FOOD_DIR_Y = 3
        ADJ_BODY_TOP = 4
        ADJ_BODY_BOTTOM = 5
        ADJ_BODY_LEFT = 6
        ADJ_BODY_RIGHT = 7
        
        # Adjoining_Wall_X
        if snake_head_x-helper.GRID_SIZE >= helper.BOARD_LIMIT_MIN:  # Near left wall
            idx[ADJ_WALL_X] = 0
        elif snake_head_x +  helper.GRID_SIZE <= helper.BOARD_LIMIT_MAX:  # Near right wall
            idx[ADJ_WALL_X] = 1
        else:  # Neither
            idx[ADJ_WALL_X] = 2

        # Adjoining_Wall_Y
        if snake_head_y-helper.GRID_SIZE >= helper.BOARD_LIMIT_MIN:  # Near top wall
            idx[ADJ_WALL_Y] = 0
        elif snake_head_y + helper.GRID_SIZE <= helper.BOARD_LIMIT_MAX:  # Near bottom wall
            idx[ADJ_WALL_Y] = 1
        else:  # Neither
            idx[ADJ_WALL_Y] = 2

        #FOOD_DIR_X
        if snake_head_x == food_x:
            idx[FOOD_DIR_X] = 0
        elif snake_head_x < food_x:
            idx[FOOD_DIR_X] = 1
        else:
            idx[FOOD_DIR_X] = 2
        
        #FOOD_DIR_Y
        if snake_head_y == food_y:
            idx[FOOD_DIR_Y] = 0
        elif snake_head_y < food_y:
            idx[FOOD_DIR_Y] = 1
        else:
            idx[FOOD_DIR_Y] = 2

        # adj body top,bottom, left, right
        idx[ADJ_BODY_TOP] = 1 if snake_head_x-helper.GRID_SIZE in snake_body_x else 0
        idx[ADJ_BODY_BOTTOM] = 1 if snake_head_x+helper.GRID_SIZE in snake_body_x else 0
        idx[ADJ_BODY_LEFT] = 1 if snake_head_y-helper.GRID_SIZE in snake_body_y else 0
        idx[ADJ_BODY_RIGHT] = 1 if snake_head_y+helper.GRID_SIZE in snake_body_y else 0   
        return tuple(idx)
        
    # Computing the reward, need not be changed.
    def compute_reward(self, points, dead):
        if dead:
            return -1
        elif points > self.points:
            return 1
        else:
            return -0.1

    #   This is the code you need to write. 
    #   This is the reinforcement learning agent
    #   use the helper_func you need to write above to
    #   decide which move is the best move that the snake needs to make 
    #   using the compute reward function defined above. 
    #   This function also keeps track of the fact that we are in 
    #   training state or testing state so that it can decide if it needs
    #   to update the Q variable. It can use the N variable to test outcomes
    #   of possible moves it can make. 
    #   the LPC variable can be used to determine the learning rate (lr), but if 
    #   you're stuck on how to do this, just use a learning rate of 0.7 first,
    #   get your code to work then work on this.
    #   gamma is another useful parameter to determine the learning rate.
    #   based on the lr, reward, and gamma values you can update the q-table.
    #   If you're not in training mode, use the q-table loaded (already done)
    #   to make moves based on that.
    #   the only thing this function should return is the best action to take
    #   ie. (0 or 1 or 2 or 3) respectively. 
    #   The parameters defined should be enough. If you want to describe more elaborate
    #   states as mentioned in helper_func, use the state variable to contain all that.
    def agent_action(self, state, points, dead):
        # print("IN AGENT_ACTION")
        s_prime = self.helper_func(state) # s'
        lr = 0.1

        def utility(state_indices):
            if random.uniform(0,1) < self.epsilon:
                return random.choice(self.actions)
            return int(np.argmax(self.Q[state_indices]))
        
        # Update Q-table after you die
        if self._train and self.s is not None:
            reward = self.compute_reward(points=points, dead=dead) * self.Ne
            self.points = points
            self.N[self.s][self.a] += 1
            # lr = 1.0 / (1.0 + self.N[self.s][self.a] / self.LPC)
            max_next_state = np.max(self.Q[s_prime])
            sample = reward + self.gamma * max_next_state 
            # Q(s,a) = (1- alpha)*Q(s,a) + alpha * sample
            self.Q[self.s][self.a] += lr * (sample - self.Q[self.s][self.a])

        if self._train:
            self.epsilon = max(0.000001, self.epsilon * .9)   
        # save last action
        action = utility(s_prime)
        self.s = s_prime
        self.a = action
        return action