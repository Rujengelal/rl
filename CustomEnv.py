import random
import gym
from gym import spaces
import numpy as np
from ismcts import Card, build_deck

from ismcts import CallBreakState
deck = [
    Card(rank, suit)
    for rank in range(2, 14 + 1)
    for suit in ["C", "D", "H", "S"]
]


def encoding(to_encode, reverse=False):
    encoded_labels = [0 for _ in range(52)]
    if to_encode is None:
        return encoded_labels

    for card in to_encode:
        try:
            idx = deck.index(card)

            encoded_labels[idx] = 1

        except:
            pass

    if not reverse:
        return encoded_labels
    encoded_labels = [1 if x == 0 else 0 for x in encoded_labels]
    return encoded_labels


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        sh = np.array([*deck, *deck, *deck])
        # print(sh.shape)
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(52)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(
            low=0, high=1, shape=sh.shape, dtype=np.uint8)

        print(Card(2, "S") in deck)

        # self.observation_space = spaces.Discrete(n=2)

    def step(self, action):
        reward = 0
        done = False
        info = {}
        observation = []

        cardToThrow = deck[action]

        validMoves = self.state.get_valid_moves(
            self.state.currentTrick, self.state.playerHands[1])

        # check if the algorithm tells to play a valid move

        if cardToThrow not in validMoves:  # not a valid move
            observation.extend(encoding(self.state.currentTrick))
            observation.extend(encoding(validMoves))
            observation.extend(encoding(self.state.discards, True))
            observation = np.array(observation).flatten().astype(np.uint8)
            reward = -100
            info = {"currentTrick": self.state.currentTrick,
                    "hands": self.state.playerHands, "played": deck[action]}  # if reward is -100 played will be the current played card
            done = True

            return observation, reward, done, info

        self.state.DoMove(random.choice([cardToThrow]))

        # simulate play till its again players turn
        while self.state.playerToMove != 1 and len(self.state.discards) < 52:

            moves = self.state.get_valid_moves(
                self.state.currentTrick, self.state.playerHands[self.state.playerToMove])
            self.state.DoMove(random.choice(moves))

        if len(self.state.discards) >= 52:
            done = True
        validMoves = self.state.get_valid_moves(
            self.state.currentTrick, self.state.playerHands[1])
        # observation = []  # current_tricks,player_hands,discards
        observation.extend(encoding(self.state.currentTrick))
        observation.extend(encoding(validMoves))
        observation.extend(encoding(self.state.discards, True))
        observation = np.array(observation).flatten().astype(np.uint8)
        reward = self.state.playerScores[1]
        info = {"currentTrick": self.state.currentTrick,
                "hands": self.state.playerHands, "played": deck[action]}  # if reward is 100/0 the played is card played before

        return observation, reward, done, info

    def reset(self):

        self.state = CallBreakState(4)
        self.state.playerToMove = 2

        while self.state.playerToMove != 1:

            moves = self.state.get_valid_moves(
                self.state.currentTrick, self.state.playerHands[self.state.playerToMove])
            self.state.DoMove(random.choice(moves))

        observation = []  # current_tricks,player_hands,discards
        # print(len(encoding(self.state.currentTrick)))
        validMoves = self.state.get_valid_moves(
            self.state.currentTrick, self.state.playerHands[1])
        observation.extend(encoding(self.state.currentTrick))
        observation.extend(encoding(validMoves))
        observation.extend(encoding(self.state.discards, True))
        observation = np.array(observation).flatten().astype(np.uint8)

        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        print("render")

    def close(self):
        print("close")
