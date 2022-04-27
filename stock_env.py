import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from readData import get_data
from tech_ind import MACD, RSI, BBP
from TabularQLearner import TabularQLearner
from backtest import assess_strategy
from OracleStrategy import OracleStrategy

class StockEnvironment:

  def __init__ (self, fixed = None, floating = None, starting_cash = None, share_limit = None):
    self.shares = share_limit
    self.fixed_cost = fixed
    self.floating_cost = floating
    self.starting_cash = starting_cash
    self.QL = None
    self.lastBuy = None


  def prepare_world (self, start_date, end_date, symbol):
    """
    Read the relevant price data and calculate some indicators.
    Return a DataFrame containing everything you need.
    """

    prices = get_data(start_date, end_date, [symbol])
    rsi = RSI(prices, 2)
    macd, signal, histogram = MACD(start_date, end_date, prices)
    bbp = BBP(start_date, end_date, prices, 14)
    prices = prices.join(rsi, how='left')
    prices = prices.join(bbp, how='left')
    prices = prices.join(macd, how='left')
    prices = prices.join(signal, how='left')
    del prices['SIG']
    return prices

  
  def calc_state(self, df, day, holdings):
    """ Quantizes the state to a single number. """

    h = 0 #flat
    if holdings > 0:
      h = 1 #long
    elif holdings < 0:
      h = 2 #short

    rsiQ1 = np.nanpercentile(df['RSI'], 20)
    rsiQ2 = np.nanpercentile(df['RSI'], 40)
    rsiQ3 = np.nanpercentile(df['RSI'], 60)
    rsiQ4 = np.nanpercentile(df['RSI'], 80)

    #bucket into 0-4
    rsiState = 0
    rsi = df.loc[day, 'RSI']
    print('calculating state for : ' + str(day))
    if (rsi < rsiQ1): rsiState = 0
    elif (rsi < rsiQ2 and rsi >= rsiQ1): rsiState = 1
    elif (rsi < rsiQ3 and rsi >= rsiQ2): rsiState = 2
    elif (rsi < rsiQ4 and rsi >= rsiQ3): rsiState = 3
    elif (rsi <= 100 and rsi >= rsiQ4): rsiState = 4

    #BBP
    bbpQ1 = np.nanpercentile(df['BBP'], 20)
    bbpQ2 = np.nanpercentile(df['BBP'], 40)
    bbpQ3 = np.nanpercentile(df['BBP'], 60)
    bbpQ4 = np.nanpercentile(df['BBP'], 80)

    #bucket into 0-4
    bbpState = 0
    bbp = df.loc[day, 'BBP']
    if (bbp < bbpQ1): bbpState = 0
    elif (bbp < bbpQ2 and bbp >= bbpQ1): bbpState = 1
    elif (bbp < bbpQ3 and bbp >= bbpQ2): bbpState = 2
    elif (bbp < bbpQ4 and bbp >= bbpQ3): bbpState = 3
    elif (bbp <= 100 and bbp >= bbpQ4): bbpState = 4

    macdQ1 = np.nanpercentile(df['MACD'], 20)
    macdQ2 = np.nanpercentile(df['MACD'], 40)
    macdQ3 = np.nanpercentile(df['MACD'], 60)
    macdQ4 = np.nanpercentile(df['MACD'], 80)

    #bucket into 0-4
    macdState = 0
    macd = df.loc[day, 'MACD']
    if (macd < macdQ1): macdState = 0
    elif (macd < macdQ2 and macd >= macdQ1): macdState = 1
    elif (macd < macdQ3 and macd >= macdQ2): macdState = 2
    elif (macd < macdQ4 and macd >= macdQ3): macdState = 3
    elif (macd <= 100 and macd >= macdQ4): macdState = 4
    
    #print("state holdings: " + str(h))
    #print("day's rsi: " + str(rsi))
    #print("state rsi: " + str(rsiState))
    #print("day's bbp: " + str(bbp))
    #print("state bbp: " + str(bbpState))
    #print("day's macd: " + str(macd))
    #print("state macd: " + str(macdState))
    
    #(1000 * h) + (100 * rsiState) + (10 * bbpState) + macdState
    s = (5**3 * h) + (5**2 * rsiState) + (5**1 * bbpState) + macdState
    #print(s)
    return s
    

  def reward(self, day, wallet, sold):

    #print('SOLD: ' + str(sold))

    r = 0
    #checking for selling isn't working

    #scenarios where
    long_term_sold_options = [(1000., 0.), (-1000., 0.),(1000., -1000.), (-1000., 1000.)]

    if (sold in long_term_sold_options and self.lastBuy != None): #sanity check
      print("giving reward for selling")
      long_term_reward = wallet.loc[day, 'Value'] - wallet.loc[self.lastBuy, 'Value']
      print('Long term reward: ' + str(long_term_reward))
      prev_day_r = wallet.loc[day, 'Value'] - wallet.shift(periods=1).loc[day,'Value']
      print('short term reward: ' + str(prev_day_r))
      r = long_term_reward + prev_day_r
    else:
      r = wallet.loc[day, 'Value'] - wallet.shift(periods=1).loc[day,'Value']
    
    return r

  def train_learner( self, start = None, end = None, symbol = None, trips = 0, dyna = 0,
                     eps = 0.0, eps_decay = 0.0 ):
    """
    Construct a Q-Learning trader and train it through many iterations of a stock
    world.  Store the trained learner in an instance variable for testing.

    Print a summary result of what happened at the end of each trip.
    Feel free to include portfolio stats or other information, but AT LEAST:

    Trip 499 net result: $13600.00
    """
    #Number of states will depend on how I quantize data
    #How to caluclate r?
    print("Initializing Learner and Preparing World...")
    self.QL = TabularQLearner(states=375, actions=3, epsilon=eps, epsilon_decay=eps_decay, dyna=100)
    data = self.prepare_world(start, end, symbol)
    wallet = pd.DataFrame(columns=['Cash', 'Holdings', 'Value', 'Trades'], index=data.index)
    #print(data)
    prevSR = 0
    endCondition = False
    tripNum = 0

    #for plotting
    srVals = []
    tVals = []

    while ((endCondition != True) and (tripNum < 50)):

      sold = (0.0, 0.0)
      tripNum += 1
      wallet['Cash'] = self.starting_cash
      wallet['Holdings'] = 0
      wallet['Value'] = 0
      wallet['Trades'] = 0

      firstDay = data.index[0]
      s = self.calc_state(data, firstDay, 0)
      a = self.QL.test(s)

      #print("first action: " + str(a))
      nextTrade = 0
      if (a == 0): #LONG
        nextTrade = 1000
        self.lastBuy = firstDay
      elif (a == 1): #FLAT
        nextTrade = 0
      elif (a == 2): #SHORT
        nextTrade = -1000
        self.lastBuy = firstDay

      wallet.loc[firstDay, 'Cash'] -= (data.loc[firstDay, symbol] * nextTrade)
      wallet.loc[firstDay, 'Holdings'] += nextTrade
      wallet.loc[firstDay, 'Value'] = wallet.loc[firstDay, 'Cash'] + (data.loc[firstDay, symbol] * wallet.loc[firstDay, 'Holdings'])
      wallet.loc[firstDay, 'Trades'] = nextTrade


      #NEED TO FACTOR IN TRADING COSTS
      for day in data.index[1:]:
        #update wallet with yesterdays values
        #wallet.loc[day, 'Holdings'] = wallet.shift(periods=1).loc[day, 'Holdings']
        wallet.loc[day, 'Cash'] = wallet.shift(periods=1).loc[day, 'Cash']
        wallet.loc[day, 'Value'] = wallet.loc[day, 'Cash'] + (data.loc[day, symbol] * wallet.shift(periods=1).loc[day, 'Holdings'])
        print(wallet)
        s = self.calc_state(data, day, wallet.shift(periods=1).loc[day, 'Holdings'])
        print("State: " + str(s))
        r = self.reward(day, wallet, sold)
        sold = (0.0,0.0)
        print("Reward: " + str(r))
        a = self.QL.train(s, r)
        print("Action: " + str(a))

        nextTrade = 0
        if (a == 0):#LONG
          if (wallet.shift(periods=1).loc[day, 'Holdings'] != 1000): 
            print('buying long position')
            nextTrade = 1000 - wallet.shift(periods=1).loc[day, 'Holdings']
            self.lastBuy = day          
        elif (a == 2):#SHORT
          if (wallet.shift(periods=1).loc[day, 'Holdings'] != -1000): 
            print('taking short position...')
            nextTrade = -1000 - wallet.shift(periods=1).loc[day, 'Holdings']
            self.lastBuy = day
        elif (a == 1): #FLAT
          print('moving to flat position...')
          nextTrade = 0 - wallet.shift(periods=1).loc[day, 'Holdings']

        #print("next Trade: " + str(nextTrade))
        cost = 0
        if nextTrade != 0:
          cost = self.fixed_cost + (self.floating_cost * nextTrade)
        wallet.loc[day, 'Cash'] -= (data.loc[day, symbol] * nextTrade) + cost
        wallet.loc[day, 'Holdings'] = wallet.shift(periods=1).loc[day, 'Holdings'] + nextTrade
        wallet.loc[day, 'Value'] = wallet.loc[day, 'Cash'] + (data.loc[day, symbol] * wallet.loc[day, 'Holdings'])
        wallet.loc[day, 'Trades'] = nextTrade

        sold = (wallet.shift(periods=1).loc[day, 'Holdings'], wallet.loc[day,'Holdings'])
      
      # Compose the output trade list.
      trade_list = []
      #print(wallet.to_string())
      for day in wallet.index:
        if wallet.loc[day,'Trades'] > 0:
          trade_list.append([day.date(), symbol, 'BUY', 1000])
        elif wallet.loc[day,'Trades'] < 0:
          trade_list.append([day.date(), symbol, 'SELL', 1000])
      
      print("Trip " + str(tripNum) + " complete!")
      #print(trade_list)
      trade_df = pd.DataFrame(trade_list, columns=['Date', 'Symbol', 'Direction', 'Shares'])
      #print(trade_df)
      trade_df.to_csv('trades.csv')

      #make call to backtester here
      stats = assess_strategy()
      if (stats[0] == prevSR):
        endCondition = True
      prevSR = stats[0]
      srVals.append(stats[4])
      tVals.append(tripNum)
      # if (tripNum % 10 == 0):
      #   plt.plot(wallet['Value'] / wallet['Value'].iloc[0])

      break
    plt.plot(tVals, srVals)
    plt.savefig('BaselineVsQTrader.png')
    return True

  
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
  print(args)
  # Create an instance of the environment class.
  env = StockEnvironment( fixed = args.fixed, floating = args.floating, starting_cash = args.cash,
                          share_limit = args.shares )

  o = OracleStrategy()
  bline = o.test(args.train_start, args.train_end)
  #plt.plot(bline / bline.iloc[0])
  print("Beginning training...")
  # Construct, train, and store a Q-learning trader.
  env.train_learner( start = args.train_start, end = args.train_end,
                     symbol = args.symbol, trips = args.trips, dyna = args.dyna,
                     eps = args.eps, eps_decay = args.eps_decay )

  # Test the learned policy and see how it does.

  # In sample.
  env.test_learner( start = args.train_start, end = args.train_end, symbol = args.symbol )

  # Out of sample.  Only do this once you are fully satisfied with the in sample performance!
  #env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol )


