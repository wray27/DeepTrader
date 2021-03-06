# DeepTrader <img src="https://github.com/wray27/DeepTrader/raw/master/Poster/DeepTrader_logo.png" width="80" height="80"/>
------------
The work from this project lead to the following research paper: [automated Creation of a High-Performing Algorithmic Trader via Deep Learning on Level-2 Limit Order Book Data](https://arxiv.org/abs/2012.00821). 

## Synopsis
For my master's thesis, I used deep learning to train an automated adaptive trader to trade on a continuous double auction that uses a limit-order-book (LOB), such as the Bristol Stock Exchange (BSE). This approach to the sales trader problem is not new. However, traders often have access to a lot more data from a LOB, which not only indicates the price, but also volume, quantities and time series information on bids and offers. 

The aim of my thesis was to research whether the utilisation of these additional parameters given by a LOB can improve the effectiveness of existing automated adaptive traders.

## [Bristol Stock Exchange](https://github.com/davecliff/BristolStockExchange)
The Bristol Stock Exchange is a simple minimal simulation of a limit-order-book financial exchange, developed for teaching. It is written in Python, is single-threaded and all in one file for ease of use by novices. This repository contains a fork of the original repository created by Dave Cliff.

## Notes on the Code
The repository was initially created for my personal use. The code inside of the repository was produced quickly to obtain results for my thesis. Addtionally, future changes to the code were not considered and the possibility of other developers were also ignored. 

However, after the publication this project has recieved a lot of attention. Therefore, **I will be making further updates to this repository**. Primarily, aiming to refactor and tidy the code for future developers.
