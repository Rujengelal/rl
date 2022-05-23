from ismcts import CallBreakState, cardToString, jsonToState, ISMCTS
from bot import get_bid, get_play_card
import os
import sys

from sanic import Sanic
from sanic.response import json
from sanic.request import Request
from sanic_cors import CORS
from stable_baselines3 import PPO
from CustomEnv import CustomEnv, getObservationSpace, deck as CARDDECK
# from joblib import load
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


def build_deck():
    cards = []
    for x in ["C", "D", "H", "S"]:
        for y in [1, 2, 3, 4, 5, 6, 7, 8, 9, "T", "J", "Q", "K"]:

            cards.append(str(y) + str(x))
    # random.shuffle(cards)
    return cards


deck = build_deck()
model = PPO.load("./model (4)",
                 custom_objects=custom_objects, verbose=1)


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
    return json({"value": "hello"})


# clf = load("./filename.joblib")


@app.route("/bid", methods=["POST"])
def bid(request: Request):
    """
    Bid is called at the starting phase of the game in callbreak.
    You will be provided with the following data:
    {
        "playerId": "P3",
        "playerIds": ["P3', "P0", "P2", "P1"].
        "cards": ["1S", "TS", "8S", "7S", "6S", "4S", "3S", "9H", "6H", "5H", "1C", "1D", "JD"],
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
                    "won": 0
                }
            }
        }
    }
    Context has metadata about the current round:
        `ruond`: the current round [1 - 5]
        `totalPoints`: the points accumulated till this round (dhoos is also subtracted)
        `bid`: bid of current round
        `won`: total hands won this round upto this point

    This is all the data that you will require for the bidding phase.
    If you feel that the data provided is insufficient, let us know in our discord server.
    """

    body = request.json
    print("Bid called")
    print(body)

    ####################################
    #     Input your code here.        #
    ####################################

    # bid = clf.predict([encoding(body["cards"])])
    # print("bid is ",bid[0])
    # return json({"value": int(bid[0])})

    # if bid[0] > 0:
    #     if bid[0] > 4:
    #         return json({"value": int(4)})

    #     return json({"value": int(bid[0])})
    # else:
    #     return json({"value": 1})

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
    # print(body)
    _cards, hands_in_play, discards = jsonToState(body)
    state = CallBreakState(4, _cards, hands_in_play,
                           discards, len(hands_in_play)+1)
    # print("***********************************")
    # print(_cards)
    # print(hands_in_play)
    # print(discards)
    validMovesPlayer = state.get_valid_moves(
        state.currentTrick, state.playerHands[1])
    obs = getObservationSpace(
        state.currentTrick, validMovesPlayer, state.discards)

    action, _state = model.predict(obs, deterministic=True)

    action = CARDDECK[action]
    print("*******************************************")
    print(action)
    print(state.playerHands[1])
    print(validMovesPlayer)
    if action in validMovesPlayer:
        return json({"value": cardToString(action)})
    else:
        return json({"value": cardToString(validMovesPlayer[0])})

    # state = state.CloneAndRandomize(len(hands_in_play)+1)
    # print(state.playerHands)

    m = ISMCTS(rootstate=state, itermax=30, verbose=False)
    # print("move is  ")
    # print(m)
    return json({"value": cardToString(m)})

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
