---
layout: post
title: Asynchronous Agent Actor Critic (A3C)
description: "Understanding Asynchronous Agent Actor Critic (A3C)"
modified: 2017-04-02
tags: [reinforcement learning, deep learning, research, paper]
image:
  feature: abstract-5.jpg
  credit: dargadgetz
  creditlink: http://www.dargadgetz.com/ios-7-abstract-wallpaper-pack-for-iphone-5-and-ipod-touch-retina/
---

I read the interesting article [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf) by the GoogleDeepMind group, and I'd like to share the insights I got from it.

![Article]({{site.url}}/assets/2017/paper.png)

The same group published the [Deep Q-learning method](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) to play Atari games with superhuman behavior some time ago. 

# Reinforcement Learning refresher

Let's briefly review what reinforcement is, and what problems it tries to solve. 

The general idea, is to have an agent that can interact in an environment by performing some action $a$. The effect of performing such action is to receive a reward $r$ and a new state $s'$, so that the cycle continues.

More formally we have:

- a discrete time indexed by $t$
- an Environment $E$, made my a set of states $\{s_t\}$
- a set of actions  $\{a_t\}$
- policy $\pi: s_t \longrightarrow a_t$, which describes what action should be taken in each state
- a reward $r_t$ after each action is performed

At each time stamp $t$, the agent will try to maximize the _future discounted reward_ defined as

$$R_t = \sum_{k =0 }^{\infty} \gamma^k r_{k+t}$$

where $\gamma \in [0,1]$ is the discount factor.
Please note that the previous sum is in reality a _finite_ sum because we assume that the episode will terminate at some point (in the case of Atari games, each game will terminate at some point, possibly for timeout).

## Approaches to Reinforcement Learning 

- Value based: optimize some value function
- Policy based: optimize the policy function
- Model based: model the environment

The problem with modeling the environment is that each different problem (e.g. Atari game), will require a different model. Value and polocy based methods are much more powerful (and interesting) because they don't require any prior knowledge of the world they are trying to model. Note that this does _not_ mean that the same (trained) model will be used with all the games. An agent that can play Breakout, is unable to play Pong and vice versa, but they can be trained from the same starting point.

# Value based: (Deep) Q-Learning

In this case we are tying to maximize the value function $Q$ defined as:

$$Q^\pi(s_t, a_t)=\mathbb E [r_{t+1}+ \gamma r_{t+2}+\gamma^2 r_{t+3} +\dots |s_t, a_t]$$

This function expresses the prediction of the future reward obtained by performing action $a_t$ in state $s_t$.
While this is a nice and sensible mathematical definition, it does not give us any practical value, as that expected value is impossible to calculate in practice. 
There is a closed for for such equation, called the _Bellman equation_:
$$Q^\pi(s_t, a_t)=\mathbb E_{s_{t+1}, a_{t+1}} [r+ \gamma Q^{\pi}(s_{t+1}, a_{t+1}) |s_{t}, a_{t}]$$.

Now if we have a small number of states and actions, we could implement $Q$ with a random initialized matrix. By using the previous equation it is possible to show that we can make the random initialized $Q$ to the real one. In case of the Atari games, considering that the state consists of 4 frames of the game, $84x84 pixels$ each the the total number of states is:

$$256^{(84 \cdot 84 \cdot 4)} \simeq 10^{67970}$$

and this assuming that each pixel is grayscale only!
Given the sparsity of the problem, a deep neural network was introduced to solve the problem.

In particular, the loss function is given by

$L(w) =\left (r+ \gamma \max_{a_{t+1}} Q(s_{t+1}, a_{t+1}, w) - Q(s_{t}, a_{t}, w)\right )^2 $

where is introduced a dependency from the weights.

## Getting the best policy out of $Q$

Given $Q$ is now easy to find the best policy for each state:

First let's define $Q^*$, the optimal value function as 

$$Q^*(s, a) = max_{\pi}Q^{\pi}(s,a) = Q^{\pi^ *}(s, a)$$

the for each state $s$ the action that maximizes the reward is given by

$$\pi^*(s) = argmax_a Q^*(s, a)$$


So  we just showed that if we are able to build $Q$, this will allow us to find the best policy, i.e. the one that will give the best reward.

Let's see another approach now: let's start with the policy function and see how we can maximize it.

# Policy based: Policy gradients

First of all we want to note that a policy can be either:

- deterministic, $a = \pi(s)$
- stochastic $\pi(a|s) = \mathbb P [a|s]$

In the first case, the same action is selected each time we are in the state $s$, while in the second case we have a distribution over all the possible states. What we really want is to to change such a distribution such that better actions are more probable and bad ones less likely.

First of all, let us introduce another function called Value function

 $$V(s_t) =\mathbb E_{\pi(s_t)}[r_{t+1} + \gamma r_{t+2} + \gamma^2 r__{t+3} + \dots ]$$

 Let's spend one minute to fully understand what's going on here.

 Given a (stochastic) policy $\pi$ over all possible actions, $V$ is the expected value of the (discounted) future reward over all the possible actions. Or: _what's the average reward I will obtain in this state given my current policy?_

 As before, we can have a closed form for $V$, that will greatly simplify our life:

$$V(s_t) = \mathbb E_{\pi(s_t)}[r_{t}+\gamma V(s_{t+1})]  $$

Side note: we can now rewrite $Q$ as:

$$Q(s, a) = r + \gamma V(s')$$

## Advantage function

If $Q$ represent the max value we could get from each state, and $V$ the average value, we can define the _advantage function_ as 

$$A(s, a)= Q(s, a) - V(s)$$

which is telling us how good is the action $a$ performed in state $s$ compared with the average. In particular

$$A>0, \text{ if } a \text{ is better than the average }$$

## Policy gradients

Consider the quantity

$$J(\pi) =\mathbb E_{\rho^{s_0}} [V(s_0)] $$

where $\rho^{s_0}$ is a distribution of starting states.
$J$ is expressing how good a policy function is. If we are able to estimate its gradient so that we can change the policy distribution and get better rewards. 

Let's now assume we are modeling $\pi$ with a deep neural networks with weights $w$, i.e. $\pi(a|s, w)$. Then the gradient of $J$ w.r.t. $w$ is

$$\nabla_\theta J(\pi) = E_{s\sim\rho^\pi,\;a\sim{\pi(s)}}[ A(s, a) \cdot \nabla_\theta\;log\;\pi(a|s) ]  $$

# Actor critic model

Let's look a the previous equation:

- $\nabla_\theta\;log\;\pi(a|s)$ is the _actor_, because it shows in which direction the (log) probability of taking action $a$ in state $s$ raises
- $A(s, a)$ is the _critic_: it's scalar that tells us what's the advantage of taking this action in this context (same action in different state will result in different rewards)

If we are not able to calculate $A$ we are done! Let's get back to the definition:

$$A(s, a) = Q(s, a) - V(s) = r + \gamma V(s') - V(s)$$

so give $V$, we are know $A$!

In practice the same neural network is approximating both the policy $\pi$ and the value function $V$ to give more stability and faster training (they are trying to model the same underlying environment so a lot of the feature are shared).

# Asynchronous

The last word of the title that we haven't explained is the asynchronous one, but it's also the easiest.

The idea is not new, and was proposed in the  Gorila (General Reinforcement Learning Architecture) framework. It is very simple and effective:

- have several agents exploring the environment at the same time (each agent owns a copy of the full environment)
- give different starting policies so that the agents are not correlated
- after a while update the global state with the contributions of each agent and restart

Interestingly the original Gorila idea was to leverage distributed systems, while in the DeepMind paper they use multiple cores from the same CPU. This comes with the advantage of saving time (and complexity) in moving data around.



