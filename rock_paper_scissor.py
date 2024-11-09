import random

# The player function which will be called to make a move
def player(prev_play: str) -> str:
    # Initialize the opponent's history on the first call
    if not hasattr(player, "history"):
        player.history = []  # This will store the sequence of the opponent's previous moves
    
    # If this is the first round (no previous move), make a random choice
    if prev_play == "":
        return random.choice(["R", "P", "S"])
    
    # Add the opponent's last move to the history
    player.history.append(prev_play)

    # Countering strategy:
    # We try to predict that the opponent may repeat their last move, so we counter it.
    if prev_play == "R":
        return "P"  # Paper beats Rock
    elif prev_play == "P":
        return "S"  # Scissors beats Paper
    elif prev_play == "S":
        return "R"  # Rock beats Scissors
    
    # In case the opponent plays randomly, we fall back to a random choice
    return random.choice(["R", "P", "S"])

