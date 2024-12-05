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

        self.epsilon = 1.0        
        self.s = None
        self.a = None

        # Create the Q and N Table to work with
        self.Q = helper.initialize_q_as_zeros()
        self.N = helper.initialize_q_as_zeros()


    #   This function sets if the program is in training mode or testing mode.
    def set_train(self):
        self._train = True

     #   This function sets if the program is in training mode or testing mode.       
    def set_eval(self):
        self._train = False

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
        print("In helper_func")
        
        idx = [0] * 8
        snake_head_x, snake_head_y = state[0], state[1]
        snake_body_x = [body[0] for body in state[2]]
        snake_body_y = [body[1] for body in state[2]]
        food_x, food_y = state[3], state[4]
        
        ADJ_WALL_X = 0
        ADJ_WALL_Y = 1
        FOOD_DIR_X = 2
        FOOD_DIR_X = 3
        ADJ_BODY_TOP = 4
        ADJ_BODY_BOTTOM = 5
        ADJ_BODY_LEFT = 6
        ADJ_BODY_RIGHT = 7
        
        # Adjoining_Wall_X
        if snake_head_x - helper.WALL_SIZE == 0 : #check the left
            idx[ADJ_WALL_X] = 0
        elif snake_head_x + 2 *(helper.WALL_SIZE) ==helper.DISPLAY_SIZE: #check the right
            idx[ADJ_WALL_X] = 1
        else: # head_x is neither 
            idx[ADJ_WALL_X] = 2
            
        #Adjoining_Wall_Y
        if snake_head_y - helper.WALL_SIZE == 0 : #check up
            idx[ADJ_WALL_Y] = 0
        elif snake_head_y + 2 *(helper.WALL_SIZE) == helper.DISPLAY_SIZE: #check down
            idx[ADJ_WALL_Y] = 1
        else: # head_y is neither
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
            idx[FOOD_DIR_X] = 0
        elif snake_head_y < food_y:
            idx[FOOD_DIR_X] = 1
        else:
            idx[FOOD_DIR_X] = 2
        # adj body top,bottom, left, right
        idx[ADJ_BODY_TOP] = True if snake_head_x-40 in snake_body_x else False
        idx[ADJ_BODY_BOTTOM] = True if snake_head_x+40 in snake_body_x else False
        idx[ADJ_BODY_LEFT] = True if snake_head_y-40 in snake_body_y else False
        idx[ADJ_BODY_RIGHT] = True if snake_head_y+40 in snake_body_y else False   
        return idx
        
        


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
        print("IN AGENT_ACTION")
        q_state = self.helper_func(state)
        action  = None
        reward = self.compute_reward(points=points, dead=dead)
                
        if self._train:
            self.N[q_state][self.a] += 1
            learning_rate = 0.7
            temp = self.Q[q_state]
            max_next_state = np.amax(temp)
            # changee to slide example
            self.Q[q_state][action] +=  learning_rate * (reward + self.gamma * max_next_state - self.Q[q_state][self.a]) 
            exploration_bonus = self.Ne / (1 + self.N[q_state])
            adjusted_q_val = self.Q[q_state] + exploration_bonus
            action = np.argmax(adjusted_q_val)
        else:
            action = np.argmax(self.Q[q_state]) # explotation
        
        # save last action
        self.s = q_state
        self.a = action
        return action 