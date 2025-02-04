## Original Code provided by Prof Axtell CSS 610 Spring 2025 Homework 2
## Code modified by L. Teinfalt 
## 02/04/25

import ZITraders
from datetime import datetime
import matplotlib.pyplot as plt

# Generate a population of agents which will be used for the model
print ("Generating Agents...")
thisRun = ZITraders.ZITraderModel()
thisRun.generateAgents()

print ("buyers")
print (thisRun.getBuyerValues())
print

print ("sellers")
print (thisRun.getSellerValues())
print

def visualization(prices, quantity):
    plt.hist(prices)
    plt.title("Histogram of Trade Prices From Trades (Single Run)")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    #plot supply and demand curve
    #quantity on x-axis 
    #prices on y-axis 
    plt.plot(quantity, label ='quantity')
    plt.plot(prices, label ='price')
    plt.title("Supply and Demand Curve")
    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

######################### First Run #########################################
# Use the generated agents to exercise the model and return the results
print ("Executing first run...")
startTime = datetime.now()
thisRun.executeTrades()
endTime = datetime.now()
print ("Model execution time (HH:MM:SS) is: " + str(endTime-startTime))
print ("Quantity traded = " + str(thisRun.getLengthTradeData()))
print ("The average price = " + str(thisRun.getAveragePriceData()) + " and the s.d. is " + str(thisRun.getStdDevPriceData()))
#create a variable to capture price data during the run
prices = thisRun.getPriceData()
#get the number of trades in this run
len = thisRun.getLengthTradeData()
print ("Number of trades in this run = " + str(len))
quantity = [i for i in range(len+1, 1, -1)]
quantity.sort(reverse = True)
prices.sort()
print("Quantities for this trade", quantity)
print("Prices for this trade", prices)
visualization(prices, quantity)
print ("1," + str(thisRun.getLengthTradeData()) + "," + str(thisRun.getAveragePriceData()) + "," + str(thisRun.getStdDevPriceData()))
print("")



######################### Second Run #########################################
# Reset the model and then run it again using the same set of agents
thisRun.resetModel()
print ("Executing second run...")
startTime = datetime.now()
thisRun.executeTrades()
endTime = datetime.now()
print ("Model execution time (HH:MM:SS) is: " + str(endTime-startTime))
print ("Quantity traded = " + str(thisRun.getLengthTradeData()))
print ("The average price = " + str(thisRun.getAveragePriceData()) + " and the s.d. is " + str(thisRun.getStdDevPriceData()))
#create a variable to capture price data during the run
prices2 = thisRun.getPriceData()
#get the number of trades in this run
len2 = thisRun.getLengthTradeData()
print ("Number of trades in this run = " + str(len2))
quantity2 = [i for i in range(len2+1, 1, -1)]
quantity2.sort(reverse = True)
prices2.sort()
print("Quantities for this trade", quantity2)
print("Prices for this trade", prices2)
visualization(prices2, quantity2)
print ("2," + str(thisRun.getLengthTradeData()) + "," + str(thisRun.getAveragePriceData()) + "," + str(thisRun.getStdDevPriceData()))



######################### 35 Runs Without Changing Agents #########################################
print ("")
print ("35 runs WITHOUT changing the agents")
print ("run,Quantity Traded, Average Price, Standard Deviation")
for i in range(35):
    # Reset the model and then run it again using the same set of agents
    thisRun.resetModel()
##    print "Executing run number " + str(i + 3)
##    startTime = datetime.now()
    thisRun.executeTrades()
    print (str(i + 3) + "," + str(thisRun.getLengthTradeData()) + "," + str(thisRun.getAveragePriceData()) + "," + str(thisRun.getStdDevPriceData()))

##################### 35 Runs With Agents #########################################
print ("35 runs changing the agents")
print ("run,Quantity Traded, Average Price, Standard Deviation")
for i in range(35):
    # Reset the model and then run it again using the same set of agents
    thisRun.resetModel()
    thisRun.generateAgents()
    thisRun.executeTrades()
    print (str(i + 1) + "," + str(thisRun.getLengthTradeData()) + "," + str(thisRun.getAveragePriceData()) + "," + str(thisRun.getStdDevPriceData()))
