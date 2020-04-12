# Trader superclass
# all Traders have a trader id, bank balance, blotter, and list of orders to execute


class Trader:

    def __init__(self, ttype, tid, balance, time):
        self.ttype = ttype      # what type / strategy this trader is
        self.tid = tid          # trader unique ID code
        self.balance = balance  # money in the bank
        self.blotter = []       # record of trades executed
        # customer orders currently being worked (fixed at 1)
        self.orders = []
        self.n_quotes = 0       # number of quotes live on LOB
        self.willing = 1        # used in ZIP etc
        self.able = 1           # used in ZIP etc
        self.birthtime = time   # used when calculating age of a trader/strategy
        self.profitpertime = 0  # profit per unit time
        self.n_trades = 0       # how many trades has this trader done?
        self.lastquote = None   # record of what its last quote was

    def __str__(self):
        return '[TID %s type %s balance %s blotter %s orders %s n_trades %s profitpertime %s]' \
                % (self.tid, self.ttype, self.balance, self.blotter, self.orders, self.n_trades, self.profitpertime)

    def add_order(self, order, verbose):
        # in this version, trader has at most one order,
        # if allow more than one, this needs to be self.orders.append(order)
        if self.n_quotes > 0:
            # this trader has a live quote on the LOB, from a previous customer order
            # need response to signal cancellation/withdrawal of that quote
            response = 'LOB_Cancel'
        else:
            response = 'Proceed'
        self.orders = [order]
        if verbose :
                print('add_order < response=%s' % response)
        return response

    def del_order(self, order):
        # this is lazy: assumes each trader has only one customer order with quantity=1, so deleting sole order
        # CHANGE TO DELETE THE HEAD OF THE LIST AND KEEP THE TAIL
        self.orders = []

    def bookkeep(self, trade, order, verbose, time):

        outstr = ""
        for order in self.orders:
                outstr = outstr + str(order)

        # add trade record to trader's blotter
        self.blotter.append(trade)
        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transactionprice = trade['price']
        if self.orders[0].otype == 'Bid':
                profit = self.orders[0].price - transactionprice
        else:
                profit = transactionprice - self.orders[0].price
        self.balance += profit
        self.n_trades += 1
        self.profitpertime = self.balance/(time - self.birthtime)
        # print (self.profitpertime)
        if profit < 0:
                print profit
                print trade
                print order
                sys.exit()

        if verbose:
                print('%s %s profit=%d balance=%d profit/time=%f' % (outstr, self.ttype, profit, self.balance, self.profitpertime))
        self.del_order(order)  # delete the order

    # specify how trader responds to events in the market
    # this is a null action, expect it to be overloaded by specific algos

    def respond(self, time, lob, trade, verbose):
            return None

    # specify how trader mutates its parameter values
    # this is a null action, expect it to be overloaded by specific algos
    def mutate(self, time, lob, trade, verbose):
            return None
