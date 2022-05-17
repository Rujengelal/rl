# This is a very simple Python 3 implementation of the Information Set Monte Carlo Tree Search algorithm.
# The function ISMCTS(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a
# state.GetRandomMove() or state.DoRandomRollout() function.
#
# An example GameState classes for Knockout Whist is included to give some idea of how you
# can write your own GameState to use ISMCTS in your hidden information game.
#
# Written by Peter Cowling, Edward Powley, Daniel Whitehouse (University of York, UK) September 2012 - August 2013.
#
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
#
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai
# Also read the article accompanying this code at ***URL HERE***

from math import sqrt, log
import random
from copy import deepcopy
import re
from typing import List
import time


def cardToString(x):
    return "??23456789TJQK1"[x.rank]+x.suit


def build_deck():
    cards = []
    for x in ["C", "D", "H", "S"]:
        for y in [1, 2, 3, 4, 5, 6, 7, 8, 9, "T", "J", "Q", "K"]:

            cards.append(str(y) + str(x))
    # random.shuffle(cards)
    return cards


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

    def __init__(self, n, playerMoves=None, currentTrick=None, discards=None, playerToMove=None):
        """Initialise the game state. n is the number of players (from 2 to 7)."""
        self.numberOfPlayers = n
        self.playerToMove = 1 if playerToMove is None else playerToMove
        self.tricksInRound = 13
        self.playerHands = {p: [] for p in range(1, self.numberOfPlayers + 1)}
        self.playerScores = {p: 0 for p in range(1, self.numberOfPlayers + 1)}
        self.discards = []         # Stores the cards that have been played already in this round
        self.currentTrick = []
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
            deck = [x for x in self.GetCardDeck() if x not in playerMoves]
            random.shuffle(deck)
            numberOfTricks = len(playerMoves)
            count = len(currentTricks)+1
            for p in range(1, self.numberOfPlayers + 1):
                if count == self.playerToMove:
                    # print("this", count)
                    self.playerHands[count] = playerMoves
                elif count > len(currentTricks):

                    self.playerHands[count] = deck[: numberOfTricks]
                    deck = deck[numberOfTricks:]
                else:
                    self.playerHands[count] = deck[: numberOfTricks-1]
                    deck = deck[numberOfTricks-1:]
                count = count % 4+1

                # if count == self.playerToMove:
                #     print("this", count)
                #     self.playerHands[count] = playerMoves
                # else:
                #     self.playerHands[count]=deck

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
        if len(spade_suit_moves) and highest_card.suit == "S":
            spade_suit_valid_moves = self.get_valid_moves(
                state, spade_suit_moves, True)

            # return sorted(spade_suit_valid_moves, key=lambda x: self.getCardWorth(x))
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
                    reward = 100
                else:
                    reward = 0

            #     self.playerScores[i]+=1*reward
                self.playerScores[i] = reward

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


class Node:
    """A node in the game tree. Note wins is always from the viewpoint of playerJustMoved."""

    def __init__(self, move=None, parent=None, playerJustMoved=None):
        self.move = move  # the move that got us to this node - "None" for the root node
        self.parentNode = parent  # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.avails = 1
        self.playerJustMoved = (
            playerJustMoved  # the only part of the state that the Node needs later
        )

    def GetUntriedMoves(self, legalMoves):
        """Return the elements of legalMoves for which this node does not have children."""

        # Find all moves for which this node *does* have children
        triedMoves = [child.move for child in self.childNodes]

        # Return all moves that are legal but have not been tried yet
        return [move for move in legalMoves if move not in triedMoves]

    def UCBSelectChild(self, legalMoves, exploration=0.7*2):
        """Use the UCB1 formula to select a child node, filtered by the given list of legal moves.
        exploration is a constant balancing between exploitation and exploration, with default value 0.7 (approximately sqrt(2) / 2)
        """

        # Filter the list of children by the list of legal moves
        legalChildren = [
            child for child in self.childNodes if child.move in legalMoves]

        # Get the child with the highest UCB score
        s = max(
            legalChildren,
            key=lambda c: float(c.wins) / float(c.visits)
            + exploration * sqrt(log(c.avails) / float(c.visits)),
        )

        # Update availability counts -- it is easier to do this now than during backpropagation
        for child in legalChildren:
            child.avails += 1

        # Return the child selected above
        return s

    def AddChild(self, m, p):
        """Add a new child node for the move m.
        Return the added child node
        """
        n = Node(move=m, parent=self, playerJustMoved=p)
        self.childNodes.append(n)
        return n

    def Update(self, terminalState):
        """Update this node - increment the visit count by one, and increase the win count by the result of terminalState for self.playerJustMoved."""
        self.visits += 1
        if self.playerJustMoved is not None:
            self.wins += terminalState.GetResult(self.playerJustMoved)

    def __repr__(self):
        return "[M:%s W/V/A: %4i/%4i/%4i]" % (
            self.move,
            self.wins,
            self.visits,
            self.avails,
        )

    def TreeToString(self, indent):
        """Represent the tree as a string, for debugging purposes."""
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
            s += c.TreeToString(indent + 1)
        return s

    def IndentString(self, indent):
        s = "\n"
        for i in range(1, indent + 1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s


def ISMCTS(rootstate, itermax, verbose=False):
    """Conduct an ISMCTS search for itermax iterations starting from rootstate.
    Return the best move from the rootstate.
    """

    rootnode = Node()

    # for i in range(itermax):
    startTime = time.time()
    endTime = startTime+itermax/1000
    while time.time() < endTime:
        node = rootnode

        # Determinize
        state = rootstate.CloneAndRandomize(rootstate.playerToMove)
        # print(state)
        # print(state.playerHands)

        # Select
        while (
            state.GetMoves() != [] and node.GetUntriedMoves(state.GetMoves()) == []
        ):  # node is fully expanded and non-terminal

            node = node.UCBSelectChild(state.GetMoves())
            state.DoMove(node.move)
            # print("selected")

        # Expand
        # print("expand")
        untriedMoves = node.GetUntriedMoves(state.GetMoves())
        # if we can expand (i.e. state/node is non-terminal)
        if untriedMoves != []:
            m = random.choice(untriedMoves)
            player = state.playerToMove
            state.DoMove(m)
            node = node.AddChild(m, player)  # add child and descend tree

        # Simulate
        # print("simulate")
        while state.GetMoves() != []:  # while state is non-terminal
            state.DoMove(random.choice(state.GetMoves()))

        # Backpropagate
        # print("backpropagate")
        while (
            node != None
        ):  # backpropagate from the expanded node and work back to the root node
            node.Update(state)
            node = node.parentNode

    # Output some information about the tree - can be omitted
    # if verbose:
    #     # print(rootnode.TreeToString(0))
    # else:
    #     # print(rootnode.ChildrenToString())

    return max(
        rootnode.childNodes, key=lambda c: c.visits
    ).move  # return the move that was most visited


def PlayGame(n, agents, state=None):
    """Play a sample game between two ISMCTS players."""
    state = state if state is not None else CallBreakState(n)
    print(state)

    while state.GetMoves() != []:
        print("beginning****************************************************")
        print(str(state))
        # Use different numbers of iterations (simulations, tree nodes) for different players
        m = agents[state.playerToMove](state)
        print("Best Move: " + str(m) + "\n")
        state.DoMove(m)

    someoneWon = False
    for p in range(1, state.numberOfPlayers + 1):
        if state.GetResult(p) > 0:
            print("Player " + str(p) + " wins!")
            someoneWon = True
    if not someoneWon:
        print("Nobody wins!")


if __name__ == "__main__":
    agents = {
        1: lambda s: ISMCTS(rootstate=s, itermax=100, verbose=False),
        2: lambda s: ISMCTS(rootstate=s, itermax=100, verbose=False),
        3: lambda s: ISMCTS(rootstate=s, itermax=100, verbose=False),
        4: lambda s: ISMCTS(rootstate=s, itermax=100, verbose=False),
    }
    PlayGame(4, agents)
    # state = CallBreakState(4)
    # a = state.get_highest_hand(
    #     [(1,Card(14, 'H')), (2,Card(14, 'S')),( 2,Card(4, 'S'))], False)
    # # a = state.get_valid_moves([], state.GetCardDeck())
    # print(a)
    # state.currentTrick = [
    #     (1, Card(14, 'H')), (2, Card(14, 'S')), (3, Card(4, 'S'))]
    # state.discards = []
    # state = state.CloneAndRandomize(1)
    # print(state.playerHands)
    # # m = ISMCTS(rootstate=state, itermax=100, verbose=False)
