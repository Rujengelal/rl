from joblib import load
from stable_baselines3 import PPO
from CustomEnv import getObservationSpace
from newBot import CallBreakState
from ismcts import cardToString, jsonToState, ISMCTS, stringToCard
from bot import get_bid, get_play_card
from mcts import mcts
from sanic_cors import CORS
from sanic.request import Request
from sanic.response import json
from sanic import Sanic
import os
import sys
import pickle
from CustomEnv import CustomEnv, getObservationSpace, deck as CARDDECK


info_set = {
    1: [0 for _ in range(52)],
    2: [0 for _ in range(52)],
    3: [0 for _ in range(52)],
    4: [0 for _ in range(52)]
}
first_iter = True

sys.setrecursionlimit(5000)
newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

custom_objects = {}
if newer_python_version:
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }


# to enable debug, run app with `DEBUG=1 python src/app.py`
DEBUG = int(os.getenv("DEBUG")) or False

app = Sanic(__name__)
CORS(app)

old_print = print
searcher = mcts(timeLimit=30000)
model = PPO.load("./src/model_new",
                 custom_objects=custom_objects, verbose=1)


def build_deck():
    cards = []
    for x in ["C", "D", "H", "S"]:
        for y in [1, 2, 3, 4, 5, 6, 7, 8, 9, "T", "J", "Q", "K"]:

            cards.append(str(y) + str(x))
    # random.shuffle(cards)
    return cards


deck = build_deck()


def print(args):
    # only log to output if in debug mode
    # logging to console is farily expensive so, log only when necessary
    if DEBUG:
        old_print(args)


def encoding(to_encode):
    encoded_labels = [0 for _ in range(53)]
    if to_encode is None:
        return encoded_labels

    for card in to_encode:
        try:
            idx = deck.index(card)

            encoded_labels[idx] = 1

        except:
            pass
    return encoded_labels


@app.route("/hi", methods=["GET"])
def hi(request: Request):
    """
    This function is required to check for the status of the server.
    When docker containers are spun, this endpoint is called continuously
    to check if the docker container is ready or not.
    Alternatively, if you need to do some pre-processing,
    do it first and then add this endpoint.
    """
    print("****************************hi")
    return json({"value": "hello"})


clf = load("./filename.joblib")


@app.route("/bid", methods=["POST"])
def bid(request: Request):
    global info_set
    global first_iter
    first_iter = True
    info_set = {
        1: [0 for _ in range(52)],
        2: [0 for _ in range(52)],
        3: [0 for _ in range(52)],
        4: [0 for _ in range(52)]
    }
    print("************************Bid called")
    body = request.json
    print(body)

    ####################################
    #     Input your code here.        #
    ####################################

    bid = clf.predict([encoding(body["cards"])])
    # print("bid is ")
    # print({"value": int(bid[0])})
    # # return json({"value": int(bid[0])})

    if bid[0] > 0:
        if bid[0] > 8:
            return json({"value": int(8)})

        return json({"value": int(bid[0])})
    else:
        return json({"value": 1})

    bid = get_bid(body["cards"])
    print(f"Returning bid: {bid}")

    # return should have a single field value which should be an int reprsenting the bid value
    return json({"value": bid})


@app.route("/play", methods=["POST"])
def play(request: Request):
    """
    Play is called at every hand of the game where the user should throw a card.
    Request data format:
    {
        "timeBudget": 1202,
        "playerId": "P1",
        "playerIds": ["P0", "P1", "P2", "P3"],
        "cards": [ "QS", "9S", "2S", "KH", "JH", "4H", "JC", "9C", "7C", "6C", "8D", "6D", "3D"],
        "played": [
            "2H",
            "8H"
        ],
        "history": [
            [1, ["TS", "KS", "1S", "5S"], 3],
            [3, ["QS", "6S", "TH", "2S"], 3],
        ],
        "context": {
            "round": 1,
            "players": {
                "P3": {
                    "totalPoints": 0,
                    "bid": 0,
                    "won": 0,
                },
                "P0": {
                    "totalPoints": 0,
                    "bid": 3,
                    "won": 0
                },
                "P2": {
                    "totalPoints": 0,
                    "bid": 3,
                    "won": 0
                },
                "P1": {
                    "totalPoints": 0,
                    "bid": 3,
                    "won": 2
                }
            }
        }
    }
    The `timeBudget` field contains the time you have left this round.
    The `played` field contins all the cards played this turn in order.
        'history` field contains an ordered list of cards played from first hand.
        Format: `start idx, [cards in clockwise order of player ids], winner idx`
            `start idx` is index of player that threw card first
            `winner idx` is index of player who won this hand
        `playerId`: own id,
        `playerIds`: list of ids in clockwise order (always same for a game)
    `context` is same as in bid. Refer to it in the bid function.

    This is all the data that you will require for the playing phase.
    If you feel that the data provided is insufficient, let us know in our discord server.
    """

    body = request.json
    # print("Play called")
    print(body)

    global info_set
    global first_iter

    _cards, hands_in_play, discards = jsonToState(body)
    if first_iter:  # initaialize information set with correct values
        for idx, card in enumerate(CARDDECK):
            playerIdx = len(hands_in_play)+1
            if card in _cards:
                for i in range(1, 5):
                    if i == playerIdx:
                        info_set[i][idx] = 100
                    else:
                        info_set[i][idx] = 0
            else:
                for i in range(1, 5):
                    if i != playerIdx:
                        info_set[i][idx] = 33
                    else:
                        info_set[i][idx] = 0
        print("*******************************************************")
        first_iter = False

    # print(info_set)

    # update information set on the basis of history
    if body["history"] != []:
        lastPlay = body["history"][-1]
        for c in lastPlay[1]:
            card = stringToCard(c)
            info_set[1][CARDDECK.index(card)] = 0
            info_set[2][CARDDECK.index(card)] = 0
            info_set[3][CARDDECK.index(card)] = 0
            info_set[4][CARDDECK.index(card)] = 0

    # update information set on the basis of not following the suits
    if hands_in_play != [] and body["history"] != []:
        lookup = {"C": 0,
                  "D": 1,
                  "H": 2,
                  "S": 3}

        # startingPlayer = body["history"][-1][2]+1
        startingPlayer = hands_in_play[0][0]
        for idx, card in hands_in_play:
            play_suit = hands_in_play[0][1].suit
            print("play_suit != card.suit")
            print(play_suit != card.suit)
            if play_suit != card.suit:
                print("suit not followed")

                startingIndx = lookup[play_suit]*13
                for c in range(startingIndx, startingIndx+13):
                    # print(play_suit)
                    # print(startingIndx)
                    # print(startingIndx+13)
                    # print(idx)
                    # print(c)
                    info_set[idx][c] = 0
                # print(info_set)

    # print(info_set)

    state = CallBreakState(4, _cards, hands_in_play,
                           discards, len(hands_in_play)+1, info_set)
    print(_cards)
    print(hands_in_play)
    print(state.playerHands)

    validMovesPlayer = state.get_valid_moves(
        state.currentTrick, state.playerHands[state.getCurrentPlayer()])
    obs = getObservationSpace(
        state.currentTrick, validMovesPlayer, state.discards)

    action, _state = model.predict(obs, deterministic=True)

    action = CARDDECK[action]
    # print("*******************************************")
    # print(action)
    # print(state.playerHands)
    # print(validMovesPlayer)
    # print(state.getCurrentPlayer())
    if action in validMovesPlayer:
        return json({"value": cardToString(action)})
    else:
        print("invalid move")
        return json({"value": cardToString(validMovesPlayer[0])})

    action = searcher.search(state)
    print({"value": str(cardToString(action.move))})
    return json({"value": str(cardToString(action.move))})

    play_card = get_play_card(
        played_str_arr=body["played"], cards_str_arr=body["cards"]
    )
    print(f"Returning play: {play_card}")

    # return should have a single field value
    # which should be an int reprsenting the index of the card to play
    #  e.g> {"value": "QS"}
    #  to play the card "QS"
    return json({"value": str(play_card)})


if __name__ == "__main__":
    # Docker image should always listen in port 7000
    app.run(host="0.0.0.0", port=7000, debug=DEBUG, access_log=DEBUG)
