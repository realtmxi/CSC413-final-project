# using Deep Q Learning Methods on blackjack Agent

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
from tqdm import tqdm
from keras.models import clone_model


class Constants:
    hit = 'hit'
    stay = 'stay'
    player1 = 'you'
    player2 = 'dealer'


class Player:
    def __init__(self):
        self._hand = []
        self._original_showing_value = 0

    def get_hand(self):
        return self._hand

    def get_action(self, state=None):
        if self.get_hand_value() < 15:
            return Constants.hit
        else:
            return Constants.stay

    def get_hand_value(self):
        return sum(self._hand)

    def get_showing_value(self):
        showing = sum(self._hand[1:])
        self._original_showing_value = showing
        return showing

    def get_original_showing_value(self):
        return self._original_showing_value

    def hit(self, deck):
        card_value = deck.draw()
        self._hand.append(card_value)

    def stay(self):
        return True

    def reset_hand(self):
        self._hand = []

    def update(self, new_state, reward):
        pass


class Learner(Player):
    def __init__(self):
        super().__init__()
        self._Q = {}
        self._last_state = None
        self._last_action = None
        self._learning_rate = .7
        self._discount = .9
        self._epsilon = .9
        self._learning = True

    def reset_hand(self):
        self._hand = []
        self._last_state = None
        self._last_action = None

    def get_action(self, state):
        if state in self._Q and np.random.uniform(0, 1) < self._epsilon:
            action = max(self._Q[state], key=self._Q[state].get)
        else:
            action = np.random.choice([Constants.hit, Constants.stay])
            if state not in self._Q:
                self._Q[state] = {}
            self._Q[state][action] = 0

        self._last_state = state
        self._last_action = action

        return action

    def update(self, new_state, reward):
        if self._learning:
            old = self._Q[self._last_state][self._last_action]

            if new_state in self._Q:
                new = self._discount * self._Q[new_state][max(self._Q[new_state], key=self._Q[new_state].get)]
            else:
                new = 0

            self._Q[self._last_state][self._last_action] = (1 - self._learning_rate) * old + self._learning_rate * (
                        reward + new)

    def get_optimal_strategy(self):
        df = pd.DataFrame(self._Q).transpose()
        df['optimal'] = df.apply(lambda x: 'hit' if x['hit'] >= x['stay'] else 'stay', axis=1)
        return df


class DQNLearner(Learner):
    def __init__(self):
        super().__init__()
        self._learning = True
        self._learning_rate = .1
        self._discount = .1
        self._epsilon = .9

        # Create Model
        model = Sequential()

        model.add(Dense(2, kernel_initializer='lecun_uniform', input_shape=(2,)))
        model.add(Activation('relu'))

        model.add(Dense(10, kernel_initializer='lecun_uniform'))
        model.add(Activation('relu'))

        model.add(Dense(4, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))

        rms = RMSprop()
        model.compile(loss='mse', optimizer=rms)

        self._model = model

    def get_action(self, state):
        rewards = self._model.predict([np.array([state])], batch_size=1, verbose=0)

        if np.random.uniform(0, 1) < self._epsilon:
            if rewards[0][0] > rewards[0][1]:
                action = Constants.hit
            else:
                action = Constants.stay
        else:
            action = np.random.choice([Constants.hit, Constants.stay])

        self._last_state = state
        self._last_action = action
        self._last_target = rewards

        return action

    def update(self, new_state, reward):
        if self._learning:
            rewards = self._model.predict([np.array([new_state])], batch_size=1, verbose=0)
            maxQ = rewards[0][0] if rewards[0][0] > rewards[0][1] else rewards[0][1]
            new = self._discount * maxQ

            if self._last_action == Constants.hit:
                self._last_target[0][0] = reward + new
            else:
                self._last_target[0][1] = reward + new

            # Update model
            self._model.fit(np.array([self._last_state]), self._last_target, batch_size=1, epochs=1, verbose=0)
            print(f"State: {self._last_state}, Action: {self._last_action}, Reward: {reward}, New State: {new_state}")

    def get_optimal_strategy(self):

        index = []
        for x in range(0, 21):
            for y in range(1, 11):
                index.append((x, y))

        df = pd.DataFrame(index=index, columns=['hit', 'stay'])

        for ind in index:
            outcome = self._model.predict([np.array([ind])], batch_size=1)
            df.loc[ind, 'hit'] = outcome[0][0]
            df.loc[ind, 'stay'] = outcome[0][1]

        df['optimal'] = df.apply(lambda x: 'hit' if x['hit'] >= x['stay'] else 'stay', axis=1)
        return df


class Game:
    def __init__(self, num_learning_rounds, learner = None, report_every=100):
        self.p = learner
        self.win = 0
        self.loss = 0
        self.game = 1
        self._num_learning_rounds = num_learning_rounds
        self._report_every = report_every

    def run(self):
        for _ in tqdm(range(self._num_learning_rounds + number_of_test_rounds), desc="Training Progress"):
            self.play_round()

    def play_round(self):
        d, p, p2, winner = self.reset_round()

        state = self.get_starting_state(p,p2)

        while True:
            p1_action = p.get_action(state)
            p2_action = p2.get_action(state)

            if p1_action == Constants.hit:
                p.hit(d)

            if p2_action == Constants.hit:
                p2.hit(d)

            if self.determine_if_bust(p):
                winner = Constants.player2
                break

            elif self.determine_if_bust(p2):
                winner = Constants.player1
                break

            if p1_action == p2_action and p1_action == Constants.stay:
                break

            state = self.get_state(p, p1_action, p2)
            p.update(state,0)

        if winner is None:
            winner = self.determine_winner(p,p2)

        if winner == Constants.player1:
            self.win += 1
            p.update(self.get_ending_state(p,p1_action,p2),1)
        else:
            self.loss += 1
            p.update(self.get_ending_state(p,p1_action,p2),-1)

        # Print round summary
        print(f"End of Game {self.game}: Winner - {winner}, Player Hand: {p.get_hand_value()}, Dealer Hand: {p2.get_hand_value()}")

        self.game += 1

        self.report()

        if self.game == self._num_learning_rounds:
            print("Turning off learning!")
            self.p._learning = False
            self.win = 0
            self.loss = 0

    def report(self):
        if self.game % self._num_learning_rounds == 0:
            print(str(self.game) +" : "  +str(self.win / (self.win + self.loss)))
        elif self.game % self._report_every == 0:
            print(str(self.win / (self.win + self.loss)))

    def get_state(self,player1,p1_action, player2):
        return (player1.get_hand_value(), player2.get_original_showing_value())

    def get_starting_state(self,player1, player2):
        return (player1.get_hand_value(), player2.get_showing_value())

    def get_ending_state(self,player1,p1_action, player2):
        return (player1.get_hand_value(), player2.get_hand_value())

    def determine_winner(self,player1,player2):
        if player1.get_hand_value() == 21 or (player1.get_hand_value() > player2.get_hand_value() and player1.get_hand_value() <= 21):
            return Constants.player1
        else:
            return Constants.player2

    def determine_if_bust(self,player):
        if player.get_hand_value() > 21:
            return True
        else:
            return False

    def reset_round(self):
        d = Deck()
        if self.p is None:
            self.p = Learner()
        else:
            self.p.reset_hand()

        p = self.p
        p2 = Player()

        winner = None
        p.hit(d)
        p2.hit(d)
        p.hit(d)
        p2.hit(d)

        return d, p, p2, winner

class Deck:
    def __init__(self):

        self.shuffle()

    def shuffle(self):
        cards = (np.arange(0,10) + 1)
        cards = np.repeat(cards,4*3) #4 suits x 3 decks
        np.random.shuffle(cards)
        self._cards = cards.tolist()

    def draw(self):
        return self._cards.pop()


class DoubleDQNLearner(DQNLearner):
    def __init__(self):
        super().__init__()
        # Clone the model for the target network
        self._target_model = clone_model(self._model)
        self._target_model.set_weights(self._model.get_weights())

    def update_target_model(self):
        """ Update the target model to match the primary model """
        self._target_model.set_weights(self._model.get_weights())

    def update(self, new_state, reward):
        if self._learning:
            # Predict Q-values for the new state using the primary model
            future_rewards = self._model.predict([np.array([new_state])], batch_size=1)

            # Select the action using the primary model but evaluate using the target model
            max_action = np.argmax(future_rewards)
            target_future_rewards = self._target_model.predict([np.array([new_state])], batch_size=1)
            maxQ = target_future_rewards[0][max_action]
            new = self._discount * maxQ

            # Update the target as before
            target = self._last_target.copy()
            if self._last_action == Constants.hit:
                target[0][0] = reward + new
            else:
                target[0][1] = reward + new

            # Update the primary model
            self._model.fit(np.array([self._last_state]), target, batch_size=1, epochs=1, verbose=0)

            # Optionally update the target model at certain intervals
            if self.game % some_interval == 0:
                self.update_target_model()


if __name__ == "__main__":
    num_learning_rounds = 20000
    game = Game(num_learning_rounds, DQNLearner())  # Deep Q Network Learner
    # game2 = Game(num_learning_rounds, Learner()) #Q learner
    # game3 = Game(num_learning_rounds, DoubleDQNLearner()) # Double Deep Q Network Learner
    number_of_test_rounds = 1000
    for k in range(0, num_learning_rounds + number_of_test_rounds):
        game.run()

    df = game.p.get_optimal_strategy()
    print(df)

    df.to_csv('../data/dqn.csv')

    # df3 = game.p.get_optimal_strategy()
    # print（df)
    # df3.to_csv('../data/double_dqn.csv')
