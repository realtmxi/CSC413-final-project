import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_card():
    card = np.random.randint(1, 14)
    return min(card, 10)


def card_value(card_id):
    return 11 if card_id == 1 else card_id


def useable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21


def total_value(hand):
    if useable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def add_card(hand, card):
    new_hand = list(hand)
    new_hand.append(card)
    return new_hand


def play_dealer(dealer_hand):
    while total_value(dealer_hand) < 17:
        dealer_hand = add_card(dealer_hand, get_card())
    return dealer_hand


def compare_hands(player_hand, dealer_hand):
    player_total = total_value(player_hand)
    dealer_total = total_value(dealer_hand)
    if player_total > 21:
        return -1
    elif dealer_total > 21:
        return 1
    elif player_total > dealer_total:
        return 1
    elif player_total < dealer_total:
        return -1
    else:
        return 0


def dp_solution():
    V = np.zeros((10, 10, 2))  # state values
    policy = np.zeros((10, 10, 2), dtype=int)  # 0: stick, 1: hit

    # value iteration
    for i in range(1000):
        old_V = np.copy(V)
        for player_total in range(12, 22):
            for dealer_card in range(1, 11):
                for ace in range(2):
                    values = []
                    # action: stick
                    dealer_hand = [dealer_card, get_card()]
                    dealer_hand = play_dealer(dealer_hand)
                    win = compare_hands([player_total], dealer_hand)
                    values.append(win)

                    # action: hit
                    if player_total < 21:
                        for card in range(1, 11):
                            new_total = player_total + card_value(card)
                            if new_total <= 21:
                                v = V[new_total - 12, dealer_card - 1, 1 if (ace or card == 1) else 0]
                                values.append(v)

                    # update policy and value
                    best_action = np.argmax(values)
                    V[player_total - 12, dealer_card - 1, ace] = values[best_action]
                    policy[player_total - 12, dealer_card - 1, ace] = best_action

        if np.max(np.abs(V - old_V)) < 1e-4:
            break

    return policy


def simulate_dp_games(policy, num_games=100000):
    wins = 0

    for _ in range(num_games):
        player_hand = [get_card(), get_card()]
        dealer_card = get_card()

        while True:
            player_total = total_value(player_hand)
            if player_total > 21 or player_total < 12:
                break
            ace = int(useable_ace(player_hand))
            action = policy[player_total - 12, dealer_card - 1, ace]
            if action == 0:
                break
            player_hand = add_card(player_hand, get_card())

        dealer_hand = [dealer_card, get_card()]
        dealer_hand = play_dealer(dealer_hand)
        result = compare_hands(player_hand, dealer_hand)
        if result == 1:
            wins += 1

    win_rate = wins / num_games
    return win_rate


if __name__ == '__main__':
    optimal_policy = dp_solution()
    win_rate = simulate_dp_games(optimal_policy)
    print(f"Win rate using DP optimal policy: {win_rate:.2f}")

