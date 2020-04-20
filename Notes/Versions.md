# Versions

## DeepTrader1.2.1

* First profitable version
* trained using ZIP, GVWY, Shaver and ZIC for 10 market sessions
* 8 neuron LSTM network 
* Learning rate = 1.5e-5 

## DeepTrader1.3

* trained using ZIP, GVWY, Shaver and Sniper for 10 market sessions

## DeepTrader1.4

* new feature added - time between consecutive trades
* trained using different combinations and ratios of four out of the different traders listed below:
    * ZIP, ZIC, GVWY, AA, GDX, SHVR, SNPR
* 80 traders - 40 buyers 40 sellers
* 12250 market sessions in total

## DeepTrader1.5

* added more metrics from the LOB, such as:
    * Smith's alpha
    * total number of orders
* still trades using limit prices
* supply and demand schedules have now been randomized to mimc the real world