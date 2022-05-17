from copy import deepcopy


class Action():
    def __init__(self, card):
        self.card = card

    def __str__(self):
        return str(self.card)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.card == other.card

    def __hash__(self):
        return hash(self.card)


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


class CallBreakState():
    def __init__(self, n, playerMoves=None, currentTrick=None, discards=None, playerToMove=None):
        self.numberOfPlayers = n
        self.playerToMove = 1 if playerToMove is None else playerToMove
        self.tricksInRound = 13
        self.playerHands = {p: [] for p in range(1, self.numberOfPlayers + 1)}
        self.playerScores = {p: 0 for p in range(1, self.numberOfPlayers + 1)}
        self.discards = []         # Stores the cards that have been played already in this round
        self.currentTrick = []

    def getCurrentPlayer(self):
        return self.playerToMove

    def getPossibleActions(self):
        possibleActions = []
        hand = self.playerHands[self.playerToMove]
        moves = self.get_valid_moves(self.currentTrick, hand)
        for i in moves:
            possibleActions.append(Action(card=i))
        return possibleActions

    def GetNextPlayer(self, p):
        return p % self.numberOfPlayers+1

    def getWinnerIndex(self):
        highestCard = self.get_highest_hand(self.currentTrick, True)
        # print("-------------------winner", self.currentTrick)

        for (i, x) in self.currentTrick:
            if x == highestCard:
                return i

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

    def takeAction(self, action):
        newState = deepcopy(self)
        newState.currentTrick.append((newState.playerToMove, action.card))
        newState.playerHands[newState.playerToMove].remove(action.card)
        newState.playerToMove = newState.GetNextPlayer(newState.playerToMove)
        if len(newState.currentTrick) >= 4:

            winnerIndex = newState.getWinnerIndex()
            newState.playerToMove = winnerIndex
            for (i, x) in newState.currentTrick:
                reward = 0
                # if i == winnerIndex:
                #     reward= 150-self.getCardWorth(x)
                # else:
                #     reward=- self.getCardWorth(x)
                if i == winnerIndex:
                    reward = 1
                else:
                    reward = -1

                newState.playerScores[i] += reward
            newState.discards += [card for (player, card)
                                  in newState.currentTrick]
            newState.currentTrick = []
            return newState

        return newState

    def isTerminal(self):
        return len(self.discards) >= 52

    def getReward(self):
        return
