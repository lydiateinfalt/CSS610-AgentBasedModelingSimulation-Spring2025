

#Import library packages
import networkx as nx
import matplotlib.pyplot as plt
import math
import random


class Agent:
    def __init__(self, agentType, value):
        # This method initializes a new model and sets the parameters
        self.buyerOrSeller = agentType
        if agentType == "Buyer":
            self.__quantityHeld = 0
            self.__value = random.randint(1, value)
        else:
            self.__quantityHeld = 1
            self.__value = random.randint(1, value)
        self.__price = 0

    def getValue(self):
        # This method returns the agent value
        return self.__value

    def getQuantityHeld(self):
        # This method returns the agent quantityHeld
        return self.__quantityHeld

    def setQuantityHeld(self, theQuantityHeld):
        # This method sets the agent quantityHeld
        self.__quantityHeld = theQuantityHeld

    def setPrice(self, thePrice):
        # This method sets the agent price
        self.__price = thePrice


class ZITraderModel:

    def __init__(self):
        # This method initializes a new model and sets the parameters
        self.__maxNumberOfTrades = 3500

        # Specify the number of agents of each type...
        self.__numberOfBuyers = 50
        self.__numberOfSellers = 50

        # Create lists to hold the price and trade data.
        self.__priceData = []
        self.__tradeData = []

        # Specify the maximum internal values...
        self.__maxBuyerValue = 50
        self.__maxSellerValue = 50

        # Create the agent fields...
        self.__buyers = []
        self.__sellers = []

        # Create Graph
        self.GBuyers = nx.Graph()
        self.GSellers = nx.Graph()

    def resetModel(self):
        # This method resets the model without changing the value or type of the previously-generated agents.
        # Reset the price and trade data
        self.__priceData = []
        self.__tradeData = []

        # Reset the buyers to having quantity 0...
        for i in range(self.__numberOfBuyers):
            self.__buyers[i].setQuantityHeld(0)
            self.__buyers[i].setPrice(0)

        # Reset the sellers to having quantity 1...
        for i in range(self.__numberOfSellers):
            self.__sellers[i].setQuantityHeld(1)
            self.__sellers[i].setPrice(0)

    def generateAgents(self):
        # This method will generate the number of agents specified within the model parameters above
        # Clear the agent fields...
        self.__buyers = []
        self.__sellers = []

        # First the buyers...
        for i in range(self.__numberOfBuyers):
            self.__buyers.append(Agent("Buyer", self.__maxBuyerValue))

        # Now the sellers...
        for i in range(self.__numberOfSellers):
            self.__sellers.append(Agent("Seller", self.__maxSellerValue))

    def executeTrades(self):
        # This method pairs agents at random and then selects a price randomly...
        for i in range(self.__maxNumberOfTrades):

            # Pick a buyer at random, then pick a 'bid' price randomly between 1 and the agent's private value
            buyer = random.choice(self.__buyers)
            bidPrice = random.randint(1, buyer.getValue())

            # Pick a seller at random, then pick an 'ask' price randomly between the agent's private value and maxSellerValue;
            seller = random.choice(self.__sellers)
            askPrice = random.randint(seller.getValue(), self.__maxBuyerValue)

            # Let's see if a deal can be made...
            if ((buyer.getQuantityHeld() == 0) and (seller.getQuantityHeld() == 1) and (bidPrice >= askPrice)):

                # First, compute the transaction price...
                transactionPrice = random.randint(askPrice, bidPrice)
                buyer.setPrice(transactionPrice)
                seller.setPrice(transactionPrice)
                self.__priceData.append(transactionPrice)

                # Then execute the exchange...
                buyer.setQuantityHeld(1)
                seller.setQuantityHeld(0)
                self.__tradeData.append(1)

    def executeTrades_parallel(self):
        # This method creates an index of buyers and sellers

        buyer_indices = [i for i, _ in enumerate(self.__buyers)]  # create a list of indexes
        seller_indices = [i for i, _ in enumerate(self.__sellers)]  # create a list of indexes
        random.shuffle(buyer_indices)
        random.shuffle(seller_indices)

        self.GBuyers.add_nodes_from(buyer_indices)
        self.GSellers.add_nodes_from(seller_indices)


        # uniform activation where all buyers are active once
        for i in buyer_indices:
            for j in seller_indices:
                buyer = self.__buyers[i]
                seller = self.__sellers[j]
                bidPrice = random.randint(1, buyer.getValue())
                askPrice = random.randint(
                    seller.getValue(), self.__maxBuyerValue)

                # Let's determine if a deal can be made
                if ((buyer.getQuantityHeld() == 0) and (seller.getQuantityHeld() == 1) and (bidPrice >= askPrice)):

                    # First, compute the transaction price...
                    transactionPrice = random.randint(askPrice, bidPrice)
                    buyer.setPrice(transactionPrice)
                    seller.setPrice(transactionPrice)
                    self.__priceData.append(transactionPrice)

                    # Then execute the exchange...
                    buyer.setQuantityHeld(1)
                    seller.setQuantityHeld(0)
                    self.__tradeData.append(1)
                    self.GBuyers.add_edge(i, j)
                    self.GSellers.add_edge(i, j)
        
        print("Start network analysis")
        self.network_analysis()

    def network_analysis(self):
      # Evaluate centrality for Buyers
      layout = nx.spring_layout(self.GBuyers)
      nx.draw(self.GBuyers, pos=layout, with_labels=True)
      plt.show()

      # Evaluate centrality for Sellers
      layout = nx.spring_layout(self.GSellers)
      nx.draw(self.GSellers, pos=layout, with_labels=True)
      plt.show()


    def getTradeData(self):
        # This method returns the tradeData
        return self.__tradeData

    def getLengthTradeData(self):
        # This method returns the number of records in the tradeData
        return len(self.__tradeData)

    def getPriceData(self):
        # This method returns the priceData
        return self.__priceData

    def getAveragePriceData(self):
        # This method returns the average of the priceData
        return float(sum(self.__priceData))/len(self.__priceData)

    def getMaxBuyerValue(self):
        # This method returns the maxBuyerValue
        return self.__maxBuyerValue

    def getMaxSellerValue(self):
        # This method returns the maxSellerValue
        return self.__maxSellerValue

    def getStdDevPriceData(self):
        # This method returns the standard deviation of the priceData
        l1 = []
        theAvg = self.getAveragePriceData()

        # Loop through each record in the list created above
        # Compute the difference between each record and the mean adding each squared value to a new list
        for i in self.__priceData:
            l1.append(math.pow(i-theAvg, 2))
        # Compute the standard deviation as the sum of the differences, divided by the number of records, square rooted
        return math.sqrt(float(sum(l1))/len(self.__priceData))

    def getBuyerValues(self):
        # This method will return the values of all of the buyers
        valueList = []
        for i in self.__buyers:
            valueList.append(i.getValue())
        return valueList

    def getSellerValues(self):
        # This method will return the values of all of the sellers
        valueList = []
        for i in self.__sellers:
            valueList.append(i.getValue())
        return valueList