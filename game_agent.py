"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math
import logging


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.

    Approach:
    ---------
        Calculate the difference between the player's available moves and two-times
    the opponent's number of available moves.
    """

    if game.is_loser(player):
        return -math.inf
    if game.is_winner(player):
        return math.inf

    # Active player's moves
    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(player_moves-2*opponent_moves)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.

    Approach:
    ---------
        Calculate the difference between the player's available moves and
        the opponent's number of available moves. Less aggressive than
        custom_score
    """
    if game.is_loser(player):
        return -math.inf
    if game.is_winner(player):
        return math.inf

    # Active player's location
    y_pl, x_pl = game.get_player_location(player)

    # Opponent's location
    y_opp, x_opp = game.get_player_location(game.get_opponent(player))

    # Maximize the distance between the player and the opponent
    return float((y_pl - y_opp)**2 + (x_pl - x_opp)**2)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.

    Approach:
    ---------
        Calculate the ratio of the player's available moves and
        the opponent's number of available moves.
    """

    if game.is_loser(player):
        return -math.inf
    if game.is_winner(player):
        return math.inf

    # Active player's moves
    player_moves = len(game.get_legal_moves(player))
    # Player lost
    if player_moves == 0:
        return -math.inf

    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    # Opponent lost
    if opponent_moves == 0:
        return math.inf

    return float(player_moves/opponent_moves)


# Not used in tournament. Only for experimenting different heuristics.
def custom_score_4(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.

    Approach:
    ---------
    Center Score
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    return float((h - y)**2 + (w - x)**2)


# Not used in tournament. Only for experimenting different heuristics.
def custom_score_5(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.

    Approach:
    ---------
        Calculate the difference between the player's available moves and two-times
    the opponent's number of available moves.
    """

    if game.is_loser(player):
        return -math.inf
    if game.is_winner(player):
        return math.inf

    # Active player's moves
    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(player_moves-opponent_moves)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=100.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax_val(self,game,depth,maximizing_player):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Available moves
        legal_moves = game.get_legal_moves()

        # If not available moves or depth = 0
        if depth == 0 or not legal_moves:
            return (self.score(game, self))

        # Init of the val based on max or min node
        val = (-math.inf) if maximizing_player else (math.inf)

        for move in legal_moves:
            # Leaf node
            next_move = game.forecast_move(move)

            next_val = self.minimax_val(next_move, depth-1, not maximizing_player)

            # Val updated based on max/min node
            val = max(val,next_val) if maximizing_player else min(val,next_val)

        return val

    # Parts of this code is adopted from:
    # https://github.com/diegoalejogm/AI-Nanodegree/tree/master/game-playing-agent
    # But it was significantly changed by Murat Senel
    def minimax(self, game, depth):

        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Available moves
        legal_moves = game.get_legal_moves()

        # If there are no available moves
        if not legal_moves:
            return (-1, -1)

        # Initialization
        # Chose a random best_move among the legal moves
        best_move = legal_moves[random.randint(0, len(legal_moves)-1)]
        best_score = float("-inf")

        val = float("-inf")
        # Root is max-none, hence the next level is a min-level
        maximizing_player = False
        for move in legal_moves:
            val = max(self.minimax_val(game.forecast_move(move), depth-1, maximizing_player), val)
            # Update best_move and best_score
            if val > best_score:
                best_score = val
                best_move = move

        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return best_move

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.

            # Iterative deepening
            iter_depth = 1
            while True:
                best_move= self.alphabeta(game, iter_depth)
                iter_depth+=1

        except SearchTimeout:
            logging.info("Search timeout")
            #pass

        # If best_move does not exist, return a random one
        # among the legal moves
        if best_move == (-1, -1):
            best_move = legal_moves[random.randint(0, len(legal_moves) - 1)]

        # Return the best move from the last completed search iteration
        return best_move


    # function MAX-VALUE based on the pseudo-code in
    # https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
    def max_val_alpha_beta(self, game, depth, alpha, beta):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #Initialization
        legal_moves, val = game.get_legal_moves(), float("-inf")

        # Exit condition
        if depth == 0 or not legal_moves:
            return (self.score(game, self))

        for move in legal_moves:
            #Leaves
            next_move = game.forecast_move(move)

            # Value of the branch
            val = max(val, self.min_val_alpha_beta(next_move, depth-1, alpha, beta))
            # Return val if >=beta
            if val >= beta:
                return val

            # Update alpha (if needed)
            alpha = max(alpha,val)

        return val

    # function MIN-VALUE based on the pseudo-code in
    # https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
    def min_val_alpha_beta(self, game, depth, alpha, beta):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #Initialization
        legal_moves, val = game.get_legal_moves(), float("inf")

        # Exit condition
        if depth == 0 or not legal_moves:
            return (self.score(game, self))

        for move in legal_moves:
            #Leaves
            next_move = game.forecast_move(move)

            # Value of the branch
            val = min(val, self.max_val_alpha_beta(next_move, depth-1, alpha, beta))
            # Return val if <=alpha
            if val <= alpha: 
                return val

            beta = min(beta,val)

        return val

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.

            (3) Root is max-node
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #List of legal_moves
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return (-1, -1)

        # Chose a random best_move among the legal moves
        best_move = legal_moves[random.randint(0, len(legal_moves)-1)]

        # Initial best_score
        best_score =  float("-inf")

        # Don't really need this but anyways
        val = float("-inf")

        for move in legal_moves:
            #Leaves
            next_game = game.forecast_move(move)

            #Root is max-node hence call min_val_alpha_beta
            val = self.min_val_alpha_beta(next_game, depth-1, alpha, beta)

            if val > best_score:
                best_score = val
                best_move = move

                if  val >= beta:
                    return best_move
                alpha = max(val, alpha)

        return best_move
