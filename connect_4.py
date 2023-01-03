import random
import time
import copy
import sys
import os
import pickle


class Connect4:
    def __init__(self, width, height, connect):
        """
        construct a connect4 board
        :param width: width of the board
        :param height: height of the board
        :param connect: number of connections needed to win
        """
        self.width = width
        self.height = height
        self.connect = connect
        self.player = "O"
        self.winner = None

        self.board = []
        for i in range(self.width):
            column = []
            for j in range(self.height):
                column.append(" ")
            self.board.append(column)

    def __str__(self):
        """
        show a representation of the board
        :return: representation of the board
        """
        # clear screen before printing the board
        os.system('cls' if os.name == 'nt' else 'clear')

        # create a string representation of the board
        board = ""
        for j in range(self.height):
            board += "---".join(["+"] * (self.width + 1)) + "\n"
            board += (
                "".join([f"| {self.board[i][::-1][j]} " for i in range(self.width)])
                + "|\n"
            )
        board += "---".join(["+"] * (self.width + 1))

        # return this string
        return board

    @classmethod
    def other_player(cls, player):
        """
        get the other player
        :param player: current player
        :return: other player
        """
        return "X" if player == "O" else "O"

    @classmethod
    def available_actions(cls, board):
        """
        return the set of allowed actions for a given board
        :param board: representation of the board state
        :return: set of available actions
        """
        # initialise set
        actions = set()
        
        # add any column with a blank space
        for i, column in enumerate(board):
            if column[-1] == " ":
                actions.add(i)
        
        # return completed set
        return actions

    def change_player(self):
        """
        update internal representation of whose turn it is
        :return: None
        """
        self.player = Connect4.other_player(self.player)

    def make_move(self, action):
        """
        update the board with an action (according to connect-4 mechanics)
        :param action: column index for piece to be "placed in"
        :return:  None
        """
        # check for errors
        if action not in Connect4.available_actions(self.board):
            raise ValueError("invalid move")
        if self.winner is not None:
            raise ValueError("game already won")

        # put tile in first blank spot
        for i, tile in enumerate(self.board[action]):
            if tile == " ":
                self.board[action][i] = self.player
                break
                
        # update player
        self.change_player()

    def check_winner(self):
        """
        check if there are self.connect of the same tile in a row vertically,
        horizontally, or diagonally - and update game.winner if so
        :return: None
        """

        # last player to move is only candidate winner
        potential_winner = Connect4.other_player(self.player)

        # keep track of how many are in a row
        connected = 0

        # orientate board/lists so that all victories can be considered
        horizontal_board = [
            [column[j] for column in self.board] for j in range(self.height)
        ]
        right_diagonal_board = [
            [
                self.board[i][index_total - i]
                for i in range(index_total + 1)
                if i in range(self.width) and index_total - i in range(self.height)
            ]
            for index_total in range(self.width + self.height - 1)
        ]
        left_diagonal_board = [
            [
                self.board[i][i - offset]
                for i in range(self.width + offset)
                if i in range(self.width) and i - offset in range(self.height)
            ]
            for offset in range(-self.height + 1, self.width)
        ]

        orientations = [
            horizontal_board,
            self.board,
            right_diagonal_board,
            left_diagonal_board,
        ]

        # check every orientation
        for board in orientations:

            # check every line/list in orientation
            for line in board:

                # check if there are enough tiles in the line to win
                player_tiles = line.count(potential_winner)
                if player_tiles < self.connect:

                    # if not move to next line
                    continue

                # check tiles from the "bottom"
                for tile in line[::-1]:

                    # reset count (of consecutive) if chain is broken
                    if tile != potential_winner:
                        player_tiles -= connected
                        connected = 0

                        # move on if there aren't still enough tiles to win
                        if player_tiles < self.connect:
                            break

                    # otherwise tile is added to the count (belongs to potential winner)
                    else:
                        connected += 1

                    # check for victory and if so update victor
                    if connected == self.connect:
                        self.winner = potential_winner
                        return


class Connect4AI:
    def __init__(self, alpha=0.5, epsilon=0.2):
        """
        initialise a reinforcement (Q-)learning agent
        :param alpha: learning rate for updating q values
        :param epsilon: exploration rate for greedy-epsilon decisions
        """
        # start with empty q learning dictionary
        self.q = dict()
        # use specified learning rate (alpha) and exploration rate (epsilon)
        self.alpha = alpha
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        """
        return or initialise the q learning values for a given (state, action) pair
        default to zero - assume equally likely to win/lose unless told otherwise
        :param state: a board state (self.board)
        :param action: a specific action available in the state
        :return: the current estimate for the utility of the given action in the given state
        """
        hashable_state = tuple([tuple(column) for column in state])
        return self.q.get((hashable_state, action), 0)

    def highest_future_reward(self, state):
        """
        calculate maximum q value for the possible resultant states
        based on available knowledge
        if there are no actions, and game is not won, return 0 (must be a draw)
        :param state: the current state
        :return: highest possible q value after next action
        """

        maximum = -float("inf")
        actions = Connect4.available_actions(state)

        if actions:
            for action in actions:
                maximum = max(self.get_q_value(state, action), maximum)
            return maximum

        return 0

    def update_q_value(self, state, action, resultant_state, reward):
        """
        update Q(s,a) taking into account the old and new estimates,
        and expected future reward.
        formally
        Q(s, a) <- Q(s,a)
                   + alpha * (new value estimate - old value estimate)
        weighing future and current rewards equally (game is deterministic)
        :param state: previous board state
        :param action: the action taken in previous state
        :param resultant_state: the current state (after action)
        :param reward: the reward of the current state (-1/0/1 for loss/draw/victory)
        :return: no return value (None)
        """
        hashable_state = tuple([tuple(column) for column in state])
        current_q_value = self.get_q_value(state, action)
        future_reward = self.highest_future_reward(resultant_state)

        self.q[(hashable_state, action)] = current_q_value + self.alpha * (
            (reward + future_reward) - current_q_value
        )

    def choose_action(self, state, epsilon=True):
        """
        choose an action using an epsilon-Greedy decision-making policy
        if epsilon is true, otherwise chose best action (based on resultant q value)
        :param state: current state
        :param epsilon: determines if decision is exploratory (greedy-epsilon) or just greedy
        :return: an available action
        """
        # make actions a sequence
        all_actions = list(Connect4.available_actions(state))

        # filter out actions without highest Q value
        best_actions = [
            action
            for action in all_actions
            if self.get_q_value(state, action) == self.highest_future_reward(state)
        ]

        # check if epsilon-greedy
        if epsilon:

            # chose if random or best based on self.epsilon (as weight)
            candidate_actions = random.choices(
                (best_actions, all_actions), [1 - self.epsilon, self.epsilon]
            )[0]

            # return a corresponding action randomly
            action = random.choice(candidate_actions)

        # otherwise use best action (greedy)
        else:
            action = random.choice(best_actions)

        # return the action
        return action


def main():

    # check for proper usage
    if len(sys.argv) not in [1, 5]:
        sys.exit(
            "Usage: python connect_4.py width height connect training_games"
        )

    # set a default if no command line arguments are given
    if len(sys.argv) == 1:
        width, height, connect, training_games = [7, 6, 4, 100000]
    else:
        # otherwise try to use given arguments
        try:
            width, height, connect, training_games = map(int, sys.argv[1:])
        except ValueError:
            sys.exit("height connect training_games must be integers")

    # get an agent according to given information
    agent = get_trained_ai(width, height, connect, training_games)

    # "coin flip" starter
    print("randomising starter...")
    time.sleep(0.5)
    human_start = random.choice([True, False])

    # play game
    play(agent, width, height, connect, human_start)


def play(agent, width, height, connect, human_start=True):
    """
    generate visuals and complete gameplay with human
    :param agent: trained AI agent (to play against)
    :param width: width of the board
    :param height: height of the board
    :param connect: number of connections needed to win
    :param human_start: whether human or AI takes the first move
    :return: None
    """
    # initialise game and human player
    game = Connect4(width, height, connect)
    human_player = "O" if human_start else "X"

    # output starter
    print("human starts!") if human_start else print("AI starts!")
    time.sleep(1)

    # set up gameplay loop
    while True:
        
        # print out the board
        print(game)

        # check if it is the player's go
        if game.player == human_player:

            # prompt user until a valid action is given
            while True:
                print("Choose column: ", end="")
                print(
                    ", ".join(
                        [
                            str(action + 1)
                            for action in Connect4.available_actions(game.board)
                        ]
                    )
                )
                try:
                    action = int(input("Column: ")) - 1
                    if action in Connect4.available_actions(game.board):
                        break
                except ValueError:
                    pass
                print("Choose a valid column")
        else:

            # otherwise inform user AI is making a decision
            print("AI making move...")
            time.sleep(1)
            action = agent.choose_action(game.board, epsilon=False)

        # perform the action and check for terminal state
        # (and print relevant information if so)
        game.make_move(action)
        game.check_winner()
        if game.winner == human_player:
            print(game)
            print("human wins!")
            break
        if game.winner == Connect4.other_player(human_player):
            print(game)
            print("AI wins!")
            break
        if not Connect4.available_actions(game.board):
            print(game)
            print("It was a draw!")
            break

        # otherwise loop should continue


def get_trained_ai(width, height, connect, training_games):
    """
    return a trained AI according to arguments
    :param width: width of the board
    :param height: height of the board
    :param connect: number of connections needed to win
    :param training_games: number of games to train on
    :return: trained AI agent
    """

    # create an agents directory if one doesn't exist
    if not os.path.isdir("agents"):
        os.mkdir("agents")

    # store candidate path
    file = os.path.join(
        "agents", f"w{width}_h{height}_c{connect}_t{training_games}.pkl"
    )

    # use model if it already exists
    if os.path.isfile(file):
        print("using previous model...")
        time.sleep(1)
        with open(file, "rb") as f:
            q_data = f.read()
            agent = pickle.loads(q_data)
        return agent

    else:

        # otherwise check if any previous model can be used to start with

        # filter for models with the same board
        models = [
            model
            for model in os.listdir("agents")
            if model.startswith(f"w{width}_h{height}_c{connect}")
        ]
        # get the training number as an integer
        t_games = [
            model.lstrip(f"w{width}_h{height}_c{connect}_t").rstrip(".pkl")
            for model in models
        ]
        t_games = list(map(int, t_games))

        # loop through training numbers (largest first)
        for t_game in sorted(t_games, reverse=True):

            # if the model is larger than the training_games required ignore it
            if t_game > training_games:
                continue

            # otherwise get model path
            starting_q = os.path.join(
                "agents", f"w{width}_h{height}_c{connect}_t{t_game}.pkl"
            )

            # retrieve agent from the file
            with open(starting_q, "rb") as f:
                print(f"loading model trained on {t_game} games...")
                time.sleep(1)
                q_data = f.read()
                agent = pickle.loads(q_data)

            # train all the extra games required using this agent
            trained_agent = train(
                width, height, connect, training_games - t_game, agent
            )

            # save the new agent
            with open(file, "wb") as f:
                print("saving agent...")
                pickle.dump(trained_agent, f)
            return trained_agent

        # no models for the same board are smaller than training_games so start from scratch
        print("training agent from scratch...")
        time.sleep(1)

        # initialise an agent and train it according to specifications
        agent = Connect4AI()
        trained_agent = train(width, height, connect, training_games, agent)

        # save this agent
        with open(file, "wb") as f:
            print("saving agent...")
            pickle.dump(trained_agent, f)
        return trained_agent


def train(width, height, connect, training_games, agent):
    """
    trains an AI agent by playing against itself
    :param width: width of board
    :param height: height of board
    :param connect: number of connections needed to win
    :param training_games: number of games to train on
    :param agent: a fresh or existing copy of the agent to train
    :return: returns trained agent
    """

    # complete training_games games
    for i in range(training_games):
        print(f"Playing game {i+1}")

        # initialise game
        game = Connect4(width, height, connect)

        # keep track of previous moves
        log_previous = {
            "O": {"state": None, "action": None},
            "X": {"state": None, "action": None},
        }

        # continually allow for moves until game is over
        while True:

            # keep track of the current state and action
            state = copy.deepcopy(game.board)
            action = agent.choose_action(state)

            # preemptively log move
            log_previous[game.player]["state"] = state
            log_previous[game.player]["action"] = action

            # make a move and change the player
            game.make_move(action)

            # keep track of new board
            resultant_state = copy.deepcopy(game.board)

            # check for a winner
            game.check_winner()

            if game.winner:
                # if there is a winner it was the last player to move

                agent.update_q_value(state, action, resultant_state, 1)

                # game player switches after move, so they are the loser
                agent.update_q_value(
                    log_previous[game.player]["state"],
                    log_previous[game.player]["action"],
                    resultant_state,
                    -1,
                )
                # game is over so break out of loop
                break
            # if there are no moves and no winner game is a draw and loop should stop
            elif not Connect4.available_actions(resultant_state):
                agent.update_q_value(state, action, resultant_state, 0)
                agent.update_q_value(
                    log_previous[game.player]["state"],
                    log_previous[game.player]["action"],
                    resultant_state,
                    0,
                )
                break

            # otherwise the previous action had no reward (didn't result in victory or defeat)
            elif log_previous[game.player]["state"]:
                agent.update_q_value(
                    log_previous[game.player]["state"],
                    log_previous[game.player]["action"],
                    resultant_state,
                    0,
                )

    # return agent after playing all games
    return agent


if __name__ == "__main__":
    main()
