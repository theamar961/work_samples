# Contextual Bandit 

## Problem Setting 

I was a member of the Enterprise Data Science team at Mutual of Omaha from May to December 2022. I would like to focus on one project in particular, which involved the implementation of the Contextual Bandit algorithm.

The process involved acquiring client data through a third-party vendor. This vendor regularly updated information about clients, prompting companies like Mutual to compete for access. This process is formally known as a First Price Sealed-Bid Auction, wherein competitors submit confidential bids, and the highest bid secures access to the client information.

Winning the auction grants access to the client's contact details, while losing means remaining unaware of the winning bid amount.

## Current Solution

The team currently was using tradional ML algorithms to send in thier bid amounts. Tradional ML is well-tested and easy to formulate, hence this was the first approach the team had taken. After noting down the historical performance of the model, the team believed that their pricing stratergy could use some improvement. Efforts were to made to identify what other solutions can be used to solve this problem. 

Improvements to make on current solution
- Competitiors would change their bidding patterns frequently, impacting the efficacy of the model. How to tackle this?
- % of leads won
- Cost per lead

## Initial Exploration - Contextual Bandit

I was tasked to research on the Contextual Bandit algorithm and understand how it can be used for our Insurance use-case. 

I spent the next few weeks learning more about bandits and  contextual bandits. I came across the Vowpal Wabbit library that simulates a Conextual Bandit. This was the primary library that I used - [Vowpal Wabbit]( https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/python_Contextual_bandits_and_Vowpal_Wabbit.html)


I understood the documentation, the different settings, parameters - what they meant and how to use them. 

## How does it work?

In the contextual bandit problem, a learner repeatedly observes a context, chooses an action, and observes a loss/cost/reward for the chosen action only. 


- You first define action spaces and context that you will be providing to the bandit
- You then provide the bandit, the context and its corresponding action space
- It picks an action to take based on the probability distribution of the action space
- Based on the cost function, the probability distribution is then updated
- Over-time the bandit converges its probability distribution to pick the most-optimal action




### Simulation Setup

I would like to show-case CB using a simulation. I believe this will give an idea of the different nuances involved in the whole setup. I am using varaibles that are defined by me in order to make the simulation experience intutive. 

#### Assumptions

- We are competing against a static bidder
- We are using epsilon-greedy exploration with Epsilon = 0.8 - we will exploit our knowledge 80% of the time, explore 20% of the time
- We are dealing with only 1 context
- There are only 5 actions to take


```python
# Exploration v Exploitation, I will talk more about this as we walk through our simulation
epsilon = 0.8


# This function defines whether the bandit is exploiting the knowledge it has, or exploring other actions
def explore_v_exploit(sample_num):
    if(sample_num < epsilon):
         return ("Exploitation")
    else:
        return ("Exploration")
        

# Assuming these are the clients, and they belong to the same segmentation
clients = [['M', 45], ['M', 43], ['F', 43], ['F', 47], ['M', 42], ['F', 44], ['M', 45]]

# This is the value set by actuaries or domain experts, 
# they believe this is the max or optimal value for a client belonging to this customer segment
actuaries_max_bid = 50


# Action space, possible bid amounts we could give to our vendor as part of the auction
#possible_actions = [bid_amount for bid_amount in range(0, actuaries_max_bid, 10)]
possible_actions = [10, 20, 30, 40 , 50]
print("Action space:", possible_actions)


# Initially all actions have the same probability to be chosen
#probability_distribution = [1/len(possible_actions)] * len(possible_actions)
probability_distribution = [0.2, 0.2, 0.2, 0.2, 0.2]
print("Probability of each action:", probability_distribution)


# This is our basic cost function, if we lose then we inccur the maximum cost.
def basic_cost_funciton(win, our_bid):
    
    if(win == False):
        cost = 1
        return cost
    
    cost = our_bid/actuaries_max_bid   
    return cost
```

    Action space: [10, 20, 30, 40, 50]
    Probability of each action: [0.2, 0.2, 0.2, 0.2, 0.2]



```python
# Assuming we are competiting against an opponent who bids the same amount for every lead, 
# we do not have access to this information, we do not know the bid amount of our opponents
static_bidder = [29, 29, 29, 29, 29, 29, 29]
```

#### Simulation 1



```python
sample_num = 0.4
print("The Contextual Bandit is currently in this stage:", explore_v_exploit(sample_num))

# Since we are exploiting, the current probability distribution will be used
probability_distribution = [0.2, 0.2, 0.2, 0.2, 0.2]

client_info = clients[0]
print("Context:", client_info)

opponent_bid = static_bidder[0]
print("Opponent bid amount:", opponent_bid)

# randomly sample an action from all the possible_actions based on the probability. Let us assume we randomly sampled 20
our_bid = 20
win = our_bid > opponent_bid

print("Our bid amount:", our_bid)
print("Did we win?:", win)
print("Cost incurred", basic_cost_funciton(win, our_bid))
```

    The Contextual Bandit is currently in this stage: Exploitation
    Context: ['M', 45]
    Opponent bid amount: 29
    Our bid amount: 20
    Did we win?: False
    Cost incurred 1


Clearly based on the above results, our model did not perform well since it lost to our opponent. Everytime the model loses, the cost function assigns it a cost of 1. We now have an updated proability distribution


```python
possible_actions = [10, 20, 30, 40, 50]

# Notice how $20 has 0 probability
probability_distribution = [0.25, 0, 0.25, 0.25, 0.25]
```

#### Simulation 2


```python
sample_num = 0.7
print("The Contextual Bandit is currently in this stage:", explore_v_exploit(sample_num))

# Since we are exploiting, the current probability distribution will be used
probability_distribution = [0.25, 0, 0.25, 0.25, 0.25]

client_info = clients[1]
print("Context:", client_info)

opponent_bid = static_bidder[1]
print("Opponent bid amount:", opponent_bid)

# randomly sample an action from all the possible_actions based on the probability. Let us assume we randomly sampled 40
our_bid = 40
win = our_bid > opponent_bid

print("Our bid amount:", our_bid)
print("Did we win?:", win)
print("Cost incurred", basic_cost_funciton(win, our_bid))
```

    The Contextual Bandit is currently in this stage: Exploitation
    Context: ['M', 43]
    Opponent bid amount: 29
    Our bid amount: 40
    Did we win?: True
    Cost incurred 0.8


The cost of this action is lesser than the previous action. This seems like an optimal action to take at this stage. So the bandit now assigns this value the highest probability. 


```python
possible_actions = [10, 20, 30, 40, 50]

# Notice how $40 has 1 probability
probability_distribution = [0, 0, 0, 1, 0]
```

#### Simulation 3


```python
sample_num = 0.1
print("The Contextual Bandit is currently in this stage:", explore_v_exploit(sample_num))

# Since we are exploiting, the current probability distribution will be used
probability_distribution = [0, 0, 0, 1, 0]

client_info = clients[2]
print("Context:", client_info)

opponent_bid = static_bidder[2]
print("Opponent bid amount:", opponent_bid)


# 40 has proability of 1, so even if we sample - we will get this value. It is exploiting the knowledge it has
our_bid = 40
win = our_bid > opponent_bid

print("Our bid amount:", our_bid)
print("Did we win?:", win)
print("Cost incurred", basic_cost_funciton(win, our_bid))
```

    The Contextual Bandit is currently in this stage: Exploitation
    Context: ['F', 43]
    Opponent bid amount: 29
    Our bid amount: 40
    Did we win?: True
    Cost incurred 0.8


#### Simulation 4


```python
sample_num = 0.23

print("The Contextual Bandit is currently in this stage:", explore_v_exploit(sample_num))
# Since we are exploiting, the current probability distribution will be used
probability_distribution = [0, 0, 0, 1, 0]


client_info = clients[3]
print("Context:", client_info)

opponent_bid = static_bidder[3]
print("Opponent bid amount:", opponent_bid)

our_bid = 40
win = our_bid > opponent_bid

print("Our bid amount:", our_bid)
print("Did we win?:", win)
print("Cost incurred", basic_cost_funciton(win, our_bid))
```

    The Contextual Bandit is currently in this stage: Exploitation
    Context: ['F', 47]
    Opponent bid amount: 29
    Our bid amount: 40
    Did we win?: True
    Cost incurred 0.8


This is where exploration v exploitation becomes necessary. Notice how the bandit is now consistently bidding the same amount.

- Is there a more optimal bid amount we have not tested?
- Is the bid amount a local minima? 
- Is this the action with the least cost?

Now let's see what happens when we explore more possible actions

#### Simulation 5


```python
sample_num = 0.9

print("The Contextual Bandit is currently in this stage:", explore_v_exploit(sample_num))

# Since we are exploring, we randomly sample from the whole action space. 
#So the proabilities are equal, regardless of the history
probability_distribution = [0.2, 0.2, 0.2, 0.2, 0.2]


client_info = clients[4]
print("Context:", client_info)

opponent_bid = static_bidder[4]
print("Opponent bid amount:", opponent_bid)


# We randomly choose 30$ during our exploration phase
our_bid = 30
win = our_bid > opponent_bid

print("Our bid amount:", our_bid)
print("Did we win?:", win)
print("Cost incurred", basic_cost_funciton(win, our_bid))
```

    The Contextual Bandit is currently in this stage: Exploration
    Context: ['M', 42]
    Opponent bid amount: 29
    Our bid amount: 30
    Did we win?: True
    Cost incurred 0.6


Action = 30$ has a lower cost, compared to the previous action. Hence the bandit now will pick this action, till it is able to find a more suitable action


```python
possible_actions = [10, 20, 30, 40, 50]

# Notice how $30 has 1 probability
probability_distribution = [0, 0, 1, 0, 0]
```

#### Simulation 6


```python
sample_num = 0.35

print("The Contextual Bandit is currently in this stage:", explore_v_exploit(sample_num))

# Since we are exploiting, the current probability distribution will be used
probability_distribution = [0, 0, 1, 0, 0]


client_info = clients[5]
print("Context:", client_info)

opponent_bid = static_bidder[5]
print("Opponent bid amount:", opponent_bid)


# We randomly choose 30$ during our exploration phase
our_bid = 30
win = our_bid > opponent_bid

print("Our bid amount:", our_bid)
print("Did we win?:", win)
print("Cost incurred", basic_cost_funciton(win, our_bid))
```

    The Contextual Bandit is currently in this stage: Exploitation
    Context: ['F', 44]
    Opponent bid amount: 29
    Our bid amount: 30
    Did we win?: True
    Cost incurred 0.6


#### Simulation 7


```python
sample_num = 0.6

print("The Contextual Bandit is currently in this stage:", explore_v_exploit(sample_num))

# Since we are exploiting, the current probability distribution will be used
probability_distribution = [0, 0, 1, 0, 0]

client_info = clients[6]
print("Context:", client_info)

opponent_bid = static_bidder[6]
print("Opponent bid amount:", opponent_bid)


# We randomly choose 30$ during our exploration phase
our_bid = 30
win = our_bid > opponent_bid

print("Our bid amount:", our_bid)
print("Did we win?:", win)
print("Cost incurred", basic_cost_funciton(win, our_bid))
```

    The Contextual Bandit is currently in this stage: Exploitation
    Context: ['M', 45]
    Opponent bid amount: 29
    Our bid amount: 30
    Did we win?: True
    Cost incurred 0.6


Action = 30$ has the lowest cost, compared to all the actions in this case. Hence the bandit now will pick this action all the time, apart from during exploration phase.

We have arrived at our most optimal price to bid!

#### AB testing

- I tested various exploration stratergies - epislon-greedy exploration, softmax exploration and explore-first exploration
- Multiple cost functions were tested with the help of advice from domain experts and team members
- In this simulation, we assume that our opponent does not change their bidding pattern. But in the real world, there are multiple types of bidders - static_bidder, random_bidder, custom_bidder. I ran the experiments across all these types of opponents as well.



#### Challenges and cons

- Vowpal Wabbit documentation is very poor, I had to dig deep to solve some of the problems I faced - stackoverflow, reddit etc
- Bandit sometimes gets stuck at a local minima, and takes a long time to arrive at the global minima
- Sometimes opponents bid more than the maximum values set by our actuaries, this means we will always lose since we have a max bid value


### Result

- Increase in % of client leads
- Decrease in cost per lead
- Able to adapt to rapidly changing/dynamic data environment - If our opponents changed their bidding pattern all of a sudden or frequently, it was able to identify this and change its parameters.


```python

```
