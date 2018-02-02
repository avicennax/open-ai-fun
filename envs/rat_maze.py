from gym.envs.toy_text import discrete

class RatMazeEnv(discrete.DiscreteEnv):
    """
    Simple rat choice maze, where a rat runs down a 
    corridor and either makes a left, where he is punished,
    or a right, where is rewarded.
    
    Maze states are as follows:
    
    (-1r) 324 (+1r)
           1
           0
        
    Where the +-1r corresponds to the reward.
    Actions are specified below.
    
    TODO: Uses matrix representation for maze.
    """
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STILL = 4
    
    C1 = 0
    C2 = 1
    C3 = 2
    LT = 3
    RT = 4
    
    def __init__(self):
        self.state = self.C1
        self.done = False
        
    def _reset(self):
        self.state = self.C1
        self.done = False
    
    def _step(self, action):
        assert not self.done
        reward = 0

        if self.state == self.C1:
            if action in [self.UP, self.STILL]:
                if action == self.UP:
                    self.state = self.C2
            else:
                self.action_error(action)
        elif self.state == self.C2:
            if action in [self.UP, self.DOWN, self.STILL]:
                if action == self.UP:
                    self.state = self.C3
                elif action == self.DOWN:
                    self.state = self.C1
            else:
                self.action_error(action)
        elif self.state == self.C3:
            if action in [self.LEFT, self.RIGHT, self.DOWN, self.STILL]:
                if action == self.LEFT:
                    reward = -1
                    self.done = True
                elif action == self.RIGHT:
                    reward = 1
                    self.done = True
                elif action == self.DOWN:
                    self.state = self.C2
            else:
                self.action_error(action)

        return self.state, reward, self.done, {}

    def action_error(self, action):
        raise ValueError("Action: {} invalid in state {}".format(action, self.state))