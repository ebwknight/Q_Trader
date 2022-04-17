import argparse
import pandas as pd
import numpy as np
from readData import get_data
from tech_ind import MACD, RSI, BBP

class StockEnvironment:

  def __init__ (self, fixed = None, floating = None, starting_cash = None, share_limit = None):
    self.shares = share_limit
    self.fixed_cost = fixed
    self.floating_cost = floating
    self.starting_cash = starting_cash


  def prepare_world (self, start_date, end_date, symbol):
    """
    Read the relevant price data and calculate some indicators.
    Return a DataFrame containing everything you need.
    """
    prices = get_data(start_date, end_date, symbol)
    rsi = RSI(prices, 2)
    macd, signal, histogram = MACD(start_date, end_date, prices)
    bbp = BBP(start_date, end_date, prices, 14)
    prices.join(rsi, how='left')
    prices.join(macd, how='left')
    prices.join(bbp, how='left')
    print(prices)

    return prices

  
  def calc_state (self, df, day, holdings):
    """ Quantizes the state to a single number. """

    pass

  
  def train_learner( self, start = None, end = None, symbol = None, trips = 0, dyna = 0,
                     eps = 0.0, eps_decay = 0.0 ):
    """
    Construct a Q-Learning trader and train it through many iterations of a stock
    world.  Store the trained learner in an instance variable for testing.

    Print a summary result of what happened at the end of each trip.
    Feel free to include portfolio stats or other information, but AT LEAST:

    Trip 499 net result: $13600.00
    """

    pass

  
  def test_learner( self, start = None, end = None, symbol = None):
    """
    Evaluate a trained Q-Learner on a particular stock trading task.

    Print a summary result of what happened during the test.
    Feel free to include portfolio stats or other information, but AT LEAST:

    Test trip, net result: $31710.00
    Benchmark result: $6690.0000
    """

    pass
  

if __name__ == '__main__':
  # Load the requested stock for the requested dates, instantiate a Q-Learning agent,
  # and let it start trading.

  parser = argparse.ArgumentParser(description='Stock environment for Q-Learning.')

  date_args = parser.add_argument_group('date arguments')
  date_args.add_argument('--train_start', default='2018-01-01', metavar='DATE', help='Start of training period.')
  date_args.add_argument('--train_end', default='2019-12-31', metavar='DATE', help='End of training period.')
  date_args.add_argument('--test_start', default='2020-01-01', metavar='DATE', help='Start of testing period.')
  date_args.add_argument('--test_end', default='2021-12-31', metavar='DATE', help='End of testing period.')

  learn_args = parser.add_argument_group('learning arguments')
  learn_args.add_argument('--dyna', default=0, type=int, help='Dyna iterations per experience.')
  learn_args.add_argument('--eps', default=0.99, type=float, metavar='EPSILON', help='Starting epsilon for epsilon-greedy.')
  learn_args.add_argument('--eps_decay', default=0.99995, type=float, metavar='DECAY', help='Decay rate for epsilon-greedy.')

  sim_args = parser.add_argument_group('simulation arguments')
  sim_args.add_argument('--cash', default=200000, type=float, help='Starting cash for the agent.')
  sim_args.add_argument('--fixed', default=0.00, type=float, help='Fixed transaction cost.')
  sim_args.add_argument('--floating', default='0.00', type=float, help='Floating transaction cost.')
  sim_args.add_argument('--shares', default=1000, type=int, help='Number of shares to trade (also position limit).')
  sim_args.add_argument('--symbol', default='DIS', help='Stock symbol to trade.')
  sim_args.add_argument('--trips', default=500, type=int, help='Round trips through training data.')

  args = parser.parse_args()


  # Create an instance of the environment class.
  env = StockEnvironment( fixed = args.fixed, floating = args.floating, starting_cash = args.cash,
                          share_limit = args.shares )

  env.prepare_world()

  # Construct, train, and store a Q-learning trader.
  env.train_learner( start = args.train_start, end = args.train_end,
                     symbol = args.symbol, trips = args.trips, dyna = args.dyna,
                     eps = args.eps, eps_decay = args.eps_decay )

  # Test the learned policy and see how it does.

  # In sample.
  env.test_learner( start = args.train_start, end = args.train_end, symbol = args.symbol )

  # Out of sample.  Only do this once you are fully satisfied with the in sample performance!
  #env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol )


