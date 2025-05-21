def get_card_value(card):
    """Converts face cards to numbers, Ace stays as 'A'."""
    if card in ['Face']:
        return 10
    elif card == 'A':
        return 'A'
    else:
        return int(card)

def is_pair(hand):
    return len(hand) == 2 and hand[0] == hand[1]

def is_soft(hand):
    return 'A' in hand and sum(get_card_value(c) if c != 'A' else 1 for c in hand) <= 10

def hand_total(hand):
    values = [get_card_value(c) for c in hand]
    total = sum(v if v != 'A' else 11 for v in values)
    if total > 21 and 'A' in hand:
        total -= 10
    return total

def blackjack_action(player_hand, dealer_card):
    dealer_val = get_card_value(dealer_card)
    total = hand_total(player_hand)

    # Handle splitting pairs
    if is_pair(player_hand):
        card = player_hand[0]
        if card in ['A', '8']:
            return 'Split'
        elif card in ['10', 'Face']:
            return 'Stand'
        elif card == '9':
            return 'Split' if dealer_val not in [7, 10, 'A'] else 'Stand'
        elif card == '7':
            return 'Split' if dealer_val <= 7 else 'Hit'
        elif card == '6':
            return 'Split' if dealer_val <= 6 else 'Hit'
        elif card == '4':
            return 'Split' if dealer_val in [5, 6] else 'Hit'
        elif card == '3' or card == '2':
            return 'Split' if dealer_val <= 7 else 'Hit'

    # Handle soft hands (contain an Ace)
    if is_soft(player_hand):
        if total >= 19:
            return 'Stand'
        elif total == 18:
            return 'Stand' if dealer_val in [2, 7, 8] else 'Hit'
        else:
            return 'Hit'

    # Handle hard hands (no usable Ace)
    if total >= 17:
        return 'Stand'
    elif 13 <= total <= 16:
        return 'Stand' if dealer_val <= 6 else 'Hit'
    elif total == 12:
        return 'Stand' if 4 <= dealer_val <= 6 else 'Hit'
    else:
        return 'Hit'
