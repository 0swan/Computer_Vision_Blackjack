def get_optimal_move(player_hand, dealer_up_card, can_split=False, can_double=False, dealer_hits_soft_17=True):
    """
    Determines the optimal Blackjack move based on the player's hand and the dealer's upcard.

    Args:
        player_hand (list of str): A list representing the player's hand (e.g., ["A", "10"]).
        dealer_up_card (str): A string representing the dealer's upcard (e.g., "7", "A").
        can_split (bool): True if the player has the option to split their hand, False otherwise.
        can_double (bool): True if the player has the option to double down, False otherwise.
        dealer_hits_soft_17 (bool): True if the dealer hits on a soft 17, False otherwise.

    Returns:
        str: The optimal move ("Hit", "Stand", "Split", "Double", or "Surrender").
             "Surrender" is included for completeness but is not always available.
    """

    player_value = calculate_hand_value_for_strategy(player_hand)

    # --- Helper function to check if a hand is a pair ---
    def is_pair(hand):
        return len(hand) == 2 and hand[0] == hand[1]

    # --- Basic Strategy Logic (Simplified) ---

    # Surrender (often only on hard 15 or 16 vs. dealer 9, 10, A)
    if player_value == 15 and dealer_up_card in ["10", "A"]:
        return "Surrender"
    elif player_value == 16 and dealer_up_card in ["9", "10", "A"]:
        return "Surrender"

    # Splitting Pairs
    if can_split and is_pair(player_hand):
        pair_rank = player_hand[0]
        if pair_rank == "A":
            return "Split"  # Always split Aces
        elif pair_rank == "8":
            return "Split"  # Always split 8s
        elif pair_rank in ["2", "3", "7"] and dealer_up_card in ["2", "3", "4", "5", "6", "7"]:
            return "Split"
        elif pair_rank in ["4"] and dealer_up_card in ["5", "6"]:
            return "Split"
        elif pair_rank in ["6"] and dealer_up_card in ["2", "3", "4", "5", "6"]:
            return "Split"
        elif pair_rank in ["9"] and dealer_up_card in ["2", "3", "4", "5", "6", "8", "9"]:
            return "Split"

    # Double Down
    if can_double:
        if player_value in [10, 11]:
            return "Double"
        elif player_value == 9 and dealer_up_card in ["3", "4", "5", "6"]:
            return "Double"
        elif player_value == 16 and "A" in player_hand and dealer_up_card in ["4", "5", "6"]: # Soft 16
            return "Double"
        elif player_value == 17 and "A" in player_hand and dealer_up_card in ["3", "4", "5", "6"]: # Soft 17
            return "Double"
        elif player_value == 18 and "A" in player_hand and dealer_up_card in ["2", "3", "4", "5", "6"]: # Soft 18
            return "Double"

    # Hit or Stand
    if player_value <= 11:
        return "Hit"
    elif player_value == 12 and dealer_up_card in ["4", "5", "6"]:
        return "Stand"
    elif player_value in [13, 14, 15, 16] and dealer_up_card in ["2", "3", "4", "5", "6"]:
        return "Stand"
    elif player_value == 17 and "A" in player_hand: # Soft 17
        return "Hit" if dealer_hits_soft_17 else "Stand"
    elif player_value > 17 and "A" in player_hand: # Soft 18, 19, 20, 21
        return "Stand"
    elif player_value >= 17:
        return "Stand"
    else:
        return "Hit" # Default to hit in ambiguous situations

def calculate_hand_value_for_strategy(hand):
    """Calculates the value of a hand for strategy purposes (Ace can be 1 or 11)."""
    value = 0
    ace_count = 0
    ranks = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 10, "Q": 10, "K": 10, "A": 11}
    for card in hand:
        rank = card
        value += ranks.get(rank, 0)
        if rank == "A":
            ace_count += 1
    while value > 21 and ace_count > 0:
        value -= 10
        ace_count -= 1
    return value

if __name__ == "__main__":
    # Example Usage (replace with your computer vision output)
    player_hand_from_cv = ["A", "7"]
    dealer_up_card_from_cv = "8"
    can_player_split = False
    can_player_double = True
    dealer_rule_hits_soft_17 = True

    optimal_action = get_optimal_move(player_hand_from_cv, dealer_up_card_from_cv, can_player_split, can_player_double, dealer_rule_hits_soft_17)
    print(f"Player's hand: {player_hand_from_cv}")
    print(f"Dealer's upcard: {dealer_up_card_from_cv}")
    print(f"Optimal move: {optimal_action}")

    player_hand_from_cv = ["10", "10"]
    dealer_up_card_from_cv = "5"
    can_player_split = True
    can_player_double = False
    dealer_rule_hits_soft_17 = False

    optimal_action = get_optimal_move(player_hand_from_cv, dealer_up_card_from_cv, can_player_split, can_player_double, dealer_rule_hits_soft_17)
    print(f"\nPlayer's hand: {player_hand_from_cv}")
    print(f"Dealer's upcard: {dealer_up_card_from_cv}")
    print(f"Optimal move: {optimal_action}")
