
from math import sqrt, log
import random
from copy import deepcopy
import re
from typing import List
import time
from mcts import mcts
from CustomEnv import deck as CARDDECK


def cardToString(x):
    return "??23456789TJQK1"[x.rank]+x.suit


def jsonToState(body):

    init_state = []
    cards = []
    my_cards = body["cards"]
    for x in my_cards:
        if x[0] == "T":
            value = 10
        elif x[0] == "J":
            value = 11
        elif x[0] == "Q":
            value = 12
        elif x[0] == "K":
            value = 13
        elif x[0] == "1":
            value = 14
        else:
            value = int(x[0])
        cards.append(Card(value, x[1]))

    for idx, x in enumerate(body["played"]):
        # value = 0
        if x[0] == "T":
            value = 10
        elif x[0] == "J":
            value = 11
        elif x[0] == "Q":
            value = 12
        elif x[0] == "K":
            value = 13
        elif x[0] == "1":
            value = 14
        else:
            value = int(x[0])

        init_state.append((idx+1, Card(value, x[1])))
    history = []
    for i in body["history"]:
        playerId = int(i[0])+1
        for x in i[1]:
            # value = 0
            if x[0] == "T":
                value = 10
            elif x[0] == "J":
                value = 11
            elif x[0] == "Q":
                value = 12
            elif x[0] == "K":
                value = 13
            elif x[0] == "1":
                value = 14
            else:
                value = int(x[0])
            history.append(Card(value, x[1]))
            playerId = playerId % 4+1

    return cards, init_state, history


class GameState:
    """A state of the game, i.e. the game board. These are the only functions which are
    absolutely necessary to implement ISMCTS in any imperfect information game,
    although they could be enhanced and made quicker, for example by using a
    GetRandomMove() function to generate a random move during rollout.
    By convention the players are numbered 1, 2, ..., self.numberOfPlayers.
    """

    def __init__(self):
        self.numberOfPlayers = 2
        self.playerToMove = 1

    def GetNextPlayer(self, p):
        """Return the player to the left of the specified player"""
        return (p % self.numberOfPlayers) + 1

    def Clone(self):
        """Create a deep clone of this game state."""
        st = GameState()
        st.playerToMove = self.playerToMove
        return st

    def CloneAndRandomize(self, observer):
        """Create a deep clone of this game state, randomizing any information not visible to the specified observer player."""
        return self.Clone()

    def DoMove(self, move):
        """Update a state by carrying out the given move.
        Must update playerToMove.
        """
        self.playerToMove = self.GetNextPlayer(self.playerToMove)

    def GetMoves(self):
        """Get all possible moves from this state."""
        pass

    def GetResult(self, player):
        """Get the game result from the viewpoint of player."""
        pass

    def __repr__(self):
        """Don't need this - but good style."""
        pass


class Card:
    """A playing card, with rank and suit.
    rank must be an integer between 2 and 14 inclusive (Jack=11, Queen=12, King=13, Ace=14)
    suit must be a string of length 1, one of 'C' (Clubs), 'D' (Diamonds), 'H' (Hearts) or 'S' (Spades)
    """

    def __init__(self, rank, suit):
        if rank not in [*range(2, 14 + 1), 0]:
            raise Exception("Invalid rank")
        if suit not in ["C", "D", "H", "S", "0"]:
            raise Exception("Invalid suit")
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return "??23456789TJQKA"[self.rank] + self.suit

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __ne__(self, other):
        return self.rank != other.rank or self.suit != other.suit


class CallBreakState(GameState):
    """A state of the game Knockout Whist.
    See http://www.pagat.com/whist/kowhist.html for a full description of the rules.
    For simplicity of implementation, this version of the game does not include the "dog's life" rule
    and the trump suit for each round is picked randomly rather than being chosen by one of the players.
    """

    def __init__(self, n, playerMoves=None, currentTrick=None, discards=None, playerToMove=None, infoSet=None):
        """Initialise the game state. n is the number of players (from 2 to 7)."""
        self.numberOfPlayers = n
        self.playerToMove = 1 if playerToMove is None else playerToMove
        self.tricksInRound = 13
        self.playerHands = {p: [] for p in range(1, self.numberOfPlayers + 1)}
        self.playerScores = {p: 0 for p in range(1, self.numberOfPlayers + 1)}
        self.discards = []         # Stores the cards that have been played already in this round
        self.currentTrick = []
        self.infoSet = infoSet
        self.Deal(playerMoves, currentTrick, discards)

    def GetNextPlayer(self, p):
        return p % self.numberOfPlayers+1

    def Clone(self):
        # print("discards are", self.discards)
        return deepcopy(self)

    def CloneAndRandomize(self, observer):
        """Create a deep clone of this game state, randomizing any information not visible to the specified observer player."""

        st = self.Clone()
        # print(st.playerHands[observer])
        # print(st.discards)
        # print([card for (player, card) in st.currentTrick])
        # The observer can see his own hand and the cards in the current trick, and can remember the cards played in previous tricks
        seenCards = (
            st.playerHands[observer]
            + st.discards
            + [card for (player, card) in st.currentTrick]
        )
        # The observer can't see the rest of the deck

        # print("*****************************", seenCards)
        unseenCards = [card for card in st.GetCardDeck()
                       if card not in seenCards]

        # Deal the unseen cards to the other players
        random.shuffle(unseenCards)
        for p in range(1, st.numberOfPlayers + 1):
            if p != observer:
                # Deal cards to player p
                # Store the size of player p's hand
                numCards = len(self.playerHands[p])
                # Give player p the first numCards unseen cards
                st.playerHands[p] = unseenCards[:numCards]
                # Remove those cards from unseenCards
                unseenCards = unseenCards[numCards:]

        return st

    def GetCardDeck(self):
        """Construct a standard deck of 52 cards."""
        return [
            Card(rank, suit)
            for rank in range(2, 14 + 1)
            for suit in ["C", "D", "H", "S"]
        ]

    def Deal(self, playerMoves=None, currentTricks=None, discards=None):
        """Reset the game state for the beginning of a new round, and deal the cards."""
        self.discards = [] if discards is None else discards
        self.currentTrick = [] if currentTricks is None else currentTricks

        # Construct a deck, shuffle it, and deal it to the players
        if playerMoves is None:
            deck = self.GetCardDeck()
            random.shuffle(deck)
            for p in range(1, self.numberOfPlayers + 1):

                self.playerHands[p] = deck[: self.tricksInRound]
                deck = deck[self.tricksInRound:]
        else:
            deck = [x for x in self.GetCardDeck(
            ) if x not in playerMoves and x not in self.discards]
            random.shuffle(deck)
            numberOfTricks = len(playerMoves)
            count = len(currentTricks)+1
            for p in range(1, self.numberOfPlayers + 1):
                if count == self.playerToMove:
                    # print("this", count)
                    self.playerHands[count] = playerMoves
                elif count > self.playerToMove:

                    # self.playerHands[count] = deck[: numberOfTricks]
                    print(self.infoSet[count])
                    self.playerHands[count], deck = self.getCardsFromInfoSet(
                        count, deck, numberOfTricks)
                    print(count)
                    print(deck)
                    # deck = deck[numberOfTricks:]
                else:
                    print(self.infoSet[count])

                    # self.playerHands[count] = deck[: numberOfTricks-1]
                    self.playerHands[count], deck = self.getCardsFromInfoSet(
                        count, deck, numberOfTricks-1)
                    print(count)

                    print(deck)

                    # deck = deck[numberOfTricks-1:]
                count = count % 4+1

    def getCardsFromInfoSet(self, p, deck, numberOfCards):
        cards = []
        tempDeck = deepcopy(deck)
        for c in deck:
            cardIndx = CARDDECK.index(c)
            if self.infoSet[p][cardIndx] != 0:
                cards.append(c)
                tempDeck.remove(c)
            if len(cards) == numberOfCards:
                break

        return cards, tempDeck

    def GetMoves(self):
        """Get all possible moves from this state."""
        hand = self.playerHands[self.playerToMove]
        # return hand

        # print("******************************", self.currentTrick)
        return self.get_valid_moves(self.currentTrick, hand)

    def get_valid_moves(self, state=None, player_hands=None, include_s=False):
        # print("state", state)
        # print("player moves", player_hands)

        highest_card = self.get_highest_hand(state, include_s)
        # print(type(highest_card))
        if highest_card == Card(0, "0"):
            return player_hands

        same_suit_cards = [
            val
            for val in player_hands
            if highest_card.suit == val.suit and val != highest_card
        ]

        same_suit_valid_cards = []
        for val in same_suit_cards:

            if self.getCardWorth(highest_card) < self.getCardWorth(val):
                same_suit_valid_cards.append(val)

        # print(same_suit_valid_cards)
        if len(same_suit_valid_cards):

            # return sorted(same_suit_valid_cards, key=lambda x: self.getCardWorth(x))
            return same_suit_valid_cards

        elif len(same_suit_cards):
            return [sorted(same_suit_cards, key=lambda x: self.getCardWorth(x))[0]]

        spade_suit_moves = [val for val in player_hands if val.suit == "S"]
        # not_suit_moves=[]
        highest_card = self.get_highest_hand(state, True)
        spade_suit_valid_moves = [x for x in spade_suit_moves if self.getCardWorth(
            highest_card) < self.getCardWorth(x)]

        if len(spade_suit_valid_moves):
            return spade_suit_valid_moves

        elif len(spade_suit_moves):
            return [sorted(spade_suit_moves, key=lambda x:self.getCardWorth(x))[0]]
        not_suit_moves = []

        not_suit_moves = [
            sorted(player_hands, key=lambda x:self.getCardWorth(x))[0]]

        return not_suit_moves

    # use include_s true for getting highest hand at end of a trick
    def getCardWorth(self, card):
        if card.suit != 'S':
            return card.rank
        else:
            return card.rank+100

    def get_highest_hand(self, hands_in_play, include_s=False):
        if len(hands_in_play) == 0:

            return Card(0, "0")
        # print(hands_in_play)
        # print("this", hands_in_play[0])

        _, highest_hand = hands_in_play[0]

        for _, i in hands_in_play:
            if (
                i.suit == highest_hand.suit or (
                    include_s or hands_in_play[0][1].suit == "S")
            ) and self.getCardWorth(i) > self.getCardWorth(highest_hand):
                highest_hand = i

        return highest_hand

    def getWinnerIndex(self):
        highestCard = self.get_highest_hand(self.currentTrick, True)
        # print("-------------------winner", self.currentTrick)

        for (i, x) in self.currentTrick:
            if x == highestCard:
                return i

    def DoMove(self, move):
        # Store the played card in the current trick
        self.currentTrick.append((self.playerToMove, move))
        # Remove the card from the player's hand
        self.playerHands[self.playerToMove].remove(move)
        # Find the next player
        self.playerToMove = self.GetNextPlayer(self.playerToMove)
        # print("Domove tricks ", self.currentTrick)

        if len(self.currentTrick) >= 4:  # end of a trick
            # Update the game state
            # print("-----------------------trick", self.currentTrick)
            # print("4 tricks")
            winnerIndex = self.getWinnerIndex()
            self.playerToMove = winnerIndex
            # print("*****************************",winnerIndex)

            # self.tricksTaken[trickWinner] += 1
            for (i, x) in self.currentTrick:
                reward = 0
                # if i == winnerIndex:
                #     reward= 150-self.getCardWorth(x)
                # else:
                #     reward=- self.getCardWorth(x)
                if i == winnerIndex:
                    reward = 10
                else:
                    reward = -5

            #     self.playerScores[i]+=1*reward
                self.playerScores[i] += reward

            # notWinners=[x for (x,_) in self.currentTrick if x !=winnerIndex]
            # self.playerScores[winnerIndex]+=150-self.getCardWorth(self.currentTrick[winnerIndex-1][1])
            # for x in notWinners:
            #     self.playerScores[winnerIndex]+=self.getCardWorth(self.currentTrick[x-1][1])
            #     self.playerScores[x]=-self.getCardWorth(self.currentTrick[x-1][1])+self.getCardWorth(self.currentTrick[winnerIndex-1][1])

            #     for y in notWinners :
            #         if y==x:
            #             continue
            #         self.playerScores[x]+=self.getCardWorth(self.currentTrick[y-1][1])

            self.discards += [card for (player, card) in self.currentTrick]
            self.currentTrick = []
        # print("---------------------", self.currentTrick)B

    def GetResult(self, player):
        # print(player, self.playerScores)
        return self.playerScores[player]

    def __repr__(self):
        """Return a human-readable representation of the state"""
        result = "Round %i" % self.tricksInRound
        result += " | P%i: " % self.playerToMove
        result += ",".join(str(card)
                           for card in self.playerHands[self.playerToMove])

        result += " | Trick: ["
        result += ",".join(
            ("%i:%s" % (player, card)) for (player, card) in self.currentTrick
        )
        result += "]"
        return result

    def getCurrentPlayer(self):
        return self.playerToMove

    def getPossibleActions(self):
        return [Action(x, self.playerToMove) for x in self.GetMoves()]

    def takeAction(self, action):
        newState = deepcopy(self)
        newState.DoMove(action.move)
        while newState.playerToMove != 1 and len(newState.discards) < 52:

            moves = newState.get_valid_moves(
                newState.currentTrick, newState.playerHands[newState.playerToMove])
            newState.DoMove(random.choice(moves))
        return newState

    def isTerminal(self):
        if len(self.discards) >= 52:
            return True
        else:
            return False

    def getReward(self):
        return self.GetResult(1)


class Action():
    def __init__(self, move, player):
        self.move = move
        self.player = player

    def __str__(self):
        return str(self.move, self.player)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.move == other.move and self.player == other.player

    def __hash__(self):
        return hash(str((self.move, self.player)))


def cardToString(x):
    return "??23456789TJQK1"[x.rank]+x.suit


if __name__ == "__main__":
    # while True:
    print("new game")
    initialState = CallBreakState(4)
    print(initialState.get_valid_moves([(1, Card(12, 'S'))], [
          Card(4, "C"), Card(10, "S"), Card(9, "D")]))
    # print(initialState.playerHands)
    # while not initialState.isTerminal():
    #     searcher = mcts(timeLimit=10)
    #     action = searcher.search(initialState=initialState)
    #     initialState = initialState.takeAction(action)

    #     # if cardToString(action.move)[0] not in [""]:
    #     # print(action)
    #     print(cardToString(action.move))
