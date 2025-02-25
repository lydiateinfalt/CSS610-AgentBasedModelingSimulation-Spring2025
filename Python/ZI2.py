## Original Code provided by Prof Axtell CSS 610 Spring 2025 Homework 2
## Code modified by L. Teinfalt 
## Calling ZI Traders 
## No Activation Scheme
## 02/21/2025

import ZITraders
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

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
# No Activation Scheme: Use the generated agents to exercise the model and return the results
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
# No Activation Scheme: Reset the model and then run it again using the same set of agents
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

#Method for visualization the number of trades, average prices, standard deviation   
def run_visualization(list1, list2, list3):
    run_list = list(range(35))
    data = {'run': run_list, "Number of Trades": list1, "Average Price": list2, "Standard Deviation": list3 }
    df = pd.DataFrame(data)

    # Plotting the data
    plt.figure(figsize=(14,7))

    # Box plot for Trades
    plt.subplot(1, 3, 1)
    plt.hist(df['Number of Trades'], bins=10, edgecolor='black')
    plt.title('Histogram: Number of Trades')
    plt.grid(True)
    plt.xlabel('Quantity Traded')

    # Plot Average Price
    plt.subplot(1, 3, 2)
    #plt.plot(df['run'], df['Average Price'], marker='o', color='orange')
    plt.boxplot(df['Average Price'])
    plt.title('Average Price over Runs')
    plt.xlabel('Run')
    plt.ylabel('Average Price')

    # Plot Standard Deviation
    plt.subplot(1, 3, 3)
    #plt.plot(df['run'], df['Standard Deviation'], marker='o', color='green')
    plt.boxplot(df['Standard Deviation'])
    plt.title('Standard Deviation over Runs')
    plt.xlabel('Run')
    plt.ylabel('Standard Deviation')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

#Model to run multiple runs specified by parameter num_runs
#Send TRUE to change_agent if new agents should be generated in ZITraders
def run_model(num_runs, change_agent):       
    print ("")
    if change_agent is True:
        print("New Agents Generated")
    else: print("Runs WITHOUT changing the agents")
    print ("Random Activation Scheme with Number of Runs ", num_runs)
    print ("run,Quantity Traded, Average Price, Standard Deviation")
    #Create empty list to hold the number of trades, average price, and the standard
    #deviation for multiple runs
    trade_count = []
    avg_price = []
    sd_price = []

    for i in range(num_runs):
        # Reset the model and then run it again using the same set of agents
        thisRun.resetModel()
        if change_agent is True:
            thisRun.generateAgents()
        ##    print "Executing run number " + str(i + 3)
        ##    startTime = datetime.now()
        thisRun.executeTrades()
        trade_num = thisRun.getLengthTradeData()
        avg_p = thisRun.getAveragePriceData()
        sd_p = thisRun.getStdDevPriceData()
        trade_count.append(trade_num)
        avg_price.append(avg_p)
        sd_price.append(sd_p)
        print (str(i + 3) + "," + str(trade_num) + "," + str(avg_p) + "," + str(sd_p))

    #At the end of runs, create visualizations
    run_visualization(trade_count, avg_price,sd_price)
    

run_model(35, False)
run_model(35, True)