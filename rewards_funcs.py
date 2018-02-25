def get_returns(rewards, gamma):
    returns = []
    # Update rewards to yield step-wise returns
    while len(rewards) > 0:
        r = rewards.pop(0) 
        returns.append(r + sum(gamma**(i + 1) * r_t for i, r_t in enumerate(rewards)))
    return returns

    
def punish_last_action(rewards, gamma, cost):
    rewards[-1] -= cost
    return get_returns(rewards, gamma)