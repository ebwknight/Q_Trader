import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pydantic import constr
from readData import get_data
from tech_ind import MACD, RSI, BBP
from TabularQLearner import TabularQLearner
from backtest import assess_strategy
from OracleStrategy import OracleStrategy

class StockEnvironment:

  def __init__ (self, fixed = 9.95, floating = 0.005, starting_cash = None, share_limit = None):
    self.shares = share_limit
    self.fixed_cost = fixed
    self.floating_cost = floating
    self.starting_cash = starting_cash
    self.QL = None
    self.lastBuy = None

    #MACD percentile buckets
    self.macdQ1 = 0
    self.macdQ2 = 0
    self.macdQ3 = 0
    self.macdQ4 = 0

    #BBP
    self.bbpQ1 = 0
    self.bbpQ2 = 0
    self.bbpQ3 = 0
    self.bbpQ4 = 0

    #RSI
    self.rsiQ1 = 0
    self.rsiQ2 = 0
    self.rsiQ3 = 0
    self.rsiQ4 = 0


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

    #MACD percentile buckets
    self.macdQ1 = np.nanpercentile(prices['MACD'], 20)
    self.macdQ2 = np.nanpercentile(prices['MACD'], 40)
    self.macdQ3 = np.nanpercentile(prices['MACD'], 60)
    self.macdQ4 = np.nanpercentile(prices['MACD'], 80)

    #BBP
    self.bbpQ1 = np.nanpercentile(prices['BBP'], 20)
    self.bbpQ2 = np.nanpercentile(prices['BBP'], 40)
    self.bbpQ3 = np.nanpercentile(prices['BBP'], 60)
    self.bbpQ4 = np.nanpercentile(prices['BBP'], 80)

    #RSI
    self.rsiQ1 = np.nanpercentile(prices['RSI'], 20)
    self.rsiQ2 = np.nanpercentile(prices['RSI'], 40)
    self.rsiQ3 = np.nanpercentile(prices['RSI'], 60)
    self.rsiQ4 = np.nanpercentile(prices['RSI'], 80)
    #print(prices)
    return prices

  
  def calc_state(self, df, day, holdings):
    """ Quantizes the state to a single number. """

    h = 0 #flat
    if holdings > 0:
      h = 1 #long
    elif holdings < 0:
      h = 2 #short

    #bucket into 0-4
    rsiState = 0
    rsi = df.loc[day, 'RSI']
    #print('calculating state for : ' + str(day))
    if (rsi < self.rsiQ1): rsiState = 0
    elif (rsi < self.rsiQ2 and rsi >= self.rsiQ1): rsiState = 1
    elif (rsi < self.rsiQ3 and rsi >= self.rsiQ2): rsiState = 2
    elif (rsi < self.rsiQ4 and rsi >= self.rsiQ3): rsiState = 3
    elif (rsi <= 100 and rsi >= self.rsiQ4): rsiState = 4

    #bucket into 0-4
    bbpState = 0
    bbp = df.loc[day, 'BBP']
    if (bbp < self.bbpQ1): bbpState = 0
    elif (bbp < self.bbpQ2 and bbp >= self.bbpQ1): bbpState = 1
    elif (bbp < self.bbpQ3 and bbp >= self.bbpQ2): bbpState = 2
    elif (bbp < self.bbpQ4 and bbp >= self.bbpQ3): bbpState = 3
    elif (bbp <= 100 and bbp >= self.bbpQ4): bbpState = 4

    #bucket into 0-4
    macdState = 0
    macd = df.loc[day, 'MACD']
    if (macd < self.macdQ1): macdState = 0
    elif (macd < self.macdQ2 and macd >= self.macdQ1): macdState = 1
    elif (macd < self.macdQ3 and macd >= self.macdQ2): macdState = 2
    elif (macd < self.macdQ4 and macd >= self.macdQ3): macdState = 3
    elif (macd <= 100 and macd >= self.macdQ4): macdState = 4
    
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

    #Short term reward
    r = wallet.loc[day, 'Value'] - wallet.shift(periods=1).loc[day,'Value']
    '''
    Return short term reward plus the value of sold.
    sold will be the cumulative value of the last long or short position
    or 0 if the position is still open or we are flat
    '''
    return r + sold

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
    self.QL = TabularQLearner(states=375, actions=3, epsilon=eps, epsilon_decay=eps_decay, dyna=50)
    data = self.prepare_world(start, end, symbol)
    wallet = pd.DataFrame(columns=['Cash', 'Holdings', 'Value', 'Trades'], index=data.index)
    #print(data)
    prevSR = 0
    endCondition = False
    tripNum = 0

    #for plotting
    srVals = []
    tVals = []
    prevFinalVal = 0

    while ((endCondition != True) and (tripNum < 200)):

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

      cost = 0
      if nextTrade != 0:
        cost = self.fixed_cost + (self.floating_cost * abs(nextTrade) * data.loc[firstDay, symbol])
      wallet.loc[firstDay, 'Cash'] -= ((data.loc[firstDay, symbol] * abs(nextTrade)) + cost)
      wallet.loc[firstDay, 'Holdings'] += nextTrade
      wallet.loc[firstDay, 'Value'] = wallet.loc[firstDay, 'Cash'] + (data.loc[firstDay, symbol] * wallet.loc[firstDay, 'Holdings'])
      wallet.loc[firstDay, 'Trades'] = nextTrade
      
      sold = 0
      for day in data.index[1:]:
        #update wallet with yesterdays values
        wallet.loc[day, 'Holdings'] = wallet.shift(periods=1).loc[day, 'Holdings']
        wallet.loc[day, 'Cash'] = wallet.shift(periods=1).loc[day, 'Cash']
        wallet.loc[day, 'Value'] = wallet.loc[day, 'Cash'] + (data.loc[day, symbol] * wallet.loc[day, 'Holdings'])
  
        s = self.calc_state(data, day, wallet.loc[day, 'Holdings'])
        #print("State: " + str(s))
        r = self.reward(day, wallet, sold)
        sold = 0
        #print("Reward: " + str(r))
        a =self.QL.train(s, r)
        #print("Action: " + str(a))
        #print(wallet)
        nextTrade = 0
        if ((a == 0) and (wallet.loc[day, 'Holdings'] != 1000)): #LONG
          #print('buying or holding long position...')
          nextTrade = 1000 - wallet.loc[day, 'Holdings']
          if (wallet.loc[day, 'Holdings'] != 0):
            sold = wallet.loc[day, 'Value'] - wallet.loc[self.lastBuy, 'Value']
          self.lastBuy = day          
        elif ((a == 2) and (wallet.loc[day, 'Holdings'] != -1000)): #SHORT
          #print('selling or holding short position...')
          nextTrade = -1000 - wallet.loc[day, 'Holdings']
          if (wallet.loc[day, 'Holdings'] != 0):
            sold = wallet.loc[day, 'Value'] - wallet.loc[self.lastBuy, 'Value']
          self.lastBuy = day
        elif (a == 1): #FLAT
          #print('moving to flat position...')
          nextTrade = 0 - wallet.loc[day, 'Holdings']
          if (wallet.loc[day, 'Holdings'] != 0):
            sold = wallet.loc[day, 'Value'] - wallet.loc[self.lastBuy, 'Value']

        #print("next Trade: " + str(nextTrade))
        cost = 0
        if nextTrade != 0:
          cost = self.fixed_cost + (self.floating_cost * abs(nextTrade) * data.loc[day, symbol])
          #print("cost " + str(cost))
        wallet.loc[day, 'Cash'] -= (data.loc[day, symbol] * nextTrade) + cost
        wallet.loc[day, 'Holdings'] += nextTrade
        #wallet.loc[day, 'Value'] = wallet.loc[day, 'Cash'] + (data.loc[day, symbol] * wallet.loc[day, 'Holdings'])
        wallet.loc[day, 'Trades'] = nextTrade

        #print(wallet)
      # Compose the output trade list.
      trade_list = []
      #print(wallet.to_string())
      for day in wallet.index:
        if wallet.loc[day,'Trades'] == 2000:
          trade_list.append([day.date(), symbol, 'BUY', 2000])
        elif wallet.loc[day,'Trades'] == 1000:
          trade_list.append([day.date(), symbol, 'BUY', 1000])
        elif wallet.loc[day,'Trades'] == -1000:
          trade_list.append([day.date(), symbol, 'SELL', 1000])
        elif wallet.loc[day,'Trades'] == -2000:
          trade_list.append([day.date(), symbol, 'SELL', 2000])
      
      print("Trip " + str(tripNum) + " complete!")
      #print(trade_list)
      trade_df = pd.DataFrame(trade_list, columns=['Date', 'Symbol', 'Direction', 'Shares'])
      trade_df = trade_df.set_index('Date')
      #print(trade_df)
      trade_df.to_csv('trades.csv')
      #print(wallet)
      #print(trade_df)
      #make call to backtester here
      #stats = assess_strategy(fixed_cost=0, floating_cost=0)
      # if (stats[0] == prevSR):
      #   endCondition = True
      # prevSR = stats[0]
      # srVals.append(stats[4])
      finalVal = wallet['Value'].iloc[-1]
      srVals.append(finalVal)
      tVals.append(tripNum)
      if (finalVal == prevFinalVal):
        endCondition = True
      prevFinalVal = finalVal
      # if (tripNum % 10 == 0):
      #   plt.plot(wallet['Value'] / wallet['Value'].iloc[0])

      #break
    stats = assess_strategy(fixed_cost=0, floating_cost=0)
    plt.plot(tVals, srVals)
    plt.savefig('QTraderTrials.png')
    return True

  
  def test_learner( self, start = None, end = None, symbol = None):
    """
    Evaluate a trained Q-Learner on a particular stock trading task.

    Print a summary result of what happened during the test.
    Feel free to include portfolio stats or other information, but AT LEAST:

    Test trip, net result: $31710.00
    Benchmark result: $6690.0000
    """
    print("\nTesting Q Learner from " + str(start) + " to " + str(end) + "\n")
    data = self.prepare_world(start, end, symbol)
    wallet = pd.DataFrame(columns=['Cash', 'Holdings', 'Value', 'Trades'], index=data.index)

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
    elif (a == 1): #FLAT
      nextTrade = 0
    elif (a == 2): #SHORT
      nextTrade = -1000

    cost = 0
    if nextTrade != 0:
      cost = self.fixed_cost + (self.floating_cost * abs(nextTrade) * data.loc[firstDay, symbol])
    wallet.loc[firstDay, 'Cash'] -= ((data.loc[firstDay, symbol] * abs(nextTrade)) + cost)
    wallet.loc[firstDay, 'Holdings'] += nextTrade
    wallet.loc[firstDay, 'Value'] = wallet.loc[firstDay, 'Cash'] + (data.loc[firstDay, symbol] * wallet.loc[firstDay, 'Holdings'])
    wallet.loc[firstDay, 'Trades'] = nextTrade
    
    for day in data.index[1:]:
      #update wallet with yesterdays values
      wallet.loc[day, 'Holdings'] = wallet.shift(periods=1).loc[day, 'Holdings']
      wallet.loc[day, 'Cash'] = wallet.shift(periods=1).loc[day, 'Cash']
      wallet.loc[day, 'Value'] = wallet.loc[day, 'Cash'] + (data.loc[day, symbol] * wallet.loc[day, 'Holdings'])

      s = self.calc_state(data, day, wallet.loc[day, 'Holdings'])
      a =self.QL.test(s)
      #print("Action: " + str(a))
      #print(wallet)
      nextTrade = 0
      if ((a == 0) and (wallet.loc[day, 'Holdings'] != 1000)): #LONG
        #print('buying or holding long position...')
        nextTrade = 1000 - wallet.loc[day, 'Holdings']
      elif ((a == 2) and (wallet.loc[day, 'Holdings'] != -1000)): #SHORT
        #print('selling or holding short position...')
        nextTrade = -1000 - wallet.loc[day, 'Holdings']
      elif (a == 1): #FLAT
        #print('moving to flat position...')
        nextTrade = 0 - wallet.loc[day, 'Holdings']

      #print("next Trade: " + str(nextTrade))
      cost = 0
      if nextTrade != 0:
        cost = self.fixed_cost + (self.floating_cost * abs(nextTrade) * data.loc[day, symbol])
        #print("cost " + str(cost))
      wallet.loc[day, 'Cash'] -= (data.loc[day, symbol] * nextTrade) + cost
      wallet.loc[day, 'Holdings'] += nextTrade
      #wallet.loc[day, 'Value'] = wallet.loc[day, 'Cash'] + (data.loc[day, symbol] * wallet.loc[day, 'Holdings'])
      wallet.loc[day, 'Trades'] = nextTrade

    trade_list = []
    #print(wallet.to_string())
    for day in wallet.index:
      if wallet.loc[day,'Trades'] == 2000:
        trade_list.append([day.date(), symbol, 'BUY', 2000])
      elif wallet.loc[day,'Trades'] == 1000:
        trade_list.append([day.date(), symbol, 'BUY', 1000])
      elif wallet.loc[day,'Trades'] == -1000:
        trade_list.append([day.date(), symbol, 'SELL', 1000])
      elif wallet.loc[day,'Trades'] == -2000:
        trade_list.append([day.date(), symbol, 'SELL', 2000])

    trade_df = pd.DataFrame(trade_list, columns=['Date', 'Symbol', 'Direction', 'Shares'])
    trade_df = trade_df.set_index('Date')
    trade_df.to_csv('trades.csv')
    #make call to backtester here
    print("Q Trader preformance: ")
    stats = assess_strategy(fixed_cost=0, floating_cost=0)
    o = OracleStrategy(start, end, [symbol])
    print("\nBaseline: ")
    bline = o.test(start_date=start, end_date=end, symbol=symbol)
    print(bline[symbol].iloc[-1])

    plt.clf()
    plt.plot(wallet.index, wallet['Value'])
    plt.plot(wallet.index, bline[symbol])
    plt.savefig('TestPerformanceVsBaseline')
    return True
  

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
  sim_args.add_argument('--fixed', default=0.0, type=float, help='Fixed transaction cost.')
  sim_args.add_argument('--floating', default='0.0', type=float, help='Floating transaction cost.')
  sim_args.add_argument('--shares', default=1000, type=int, help='Number of shares to trade (also position limit).')
  sim_args.add_argument('--symbol', default='DIS', help='Stock symbol to trade.')
  sim_args.add_argument('--trips', default=500, type=int, help='Round trips through training data.')

  args = parser.parse_args()
  print(args)
  # Create an instance of the environment class.
  env = StockEnvironment( fixed = args.fixed, floating = args.floating, starting_cash = args.cash,
                          share_limit = args.shares )

  #o = OracleStrategy()
  #bline = o.test(args.train_start, args.train_end)
  #plt.plot(bline / bline.iloc[0])
  print("Beginning training...")
  # Construct, train, and store a Q-learning trader.
  env.train_learner( start = args.train_start, end = args.train_end,
                     symbol = args.symbol, trips = args.trips, dyna = args.dyna,
                     eps = args.eps, eps_decay = args.eps_decay )

  # Test the learned policy and see how it does.

  # In sample.
  env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol )

  # Out of sample.  Only do this once you are fully satisfied with the in sample performance!
  #env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol )

