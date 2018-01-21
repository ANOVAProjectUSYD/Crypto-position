# Cryptocurrency Technical Analysis
This project seeks to scrape Cryptocurrency data from [coinmarketcap](https://coinmarketcap.com/) and into a CSV file. Then, we will run some technical analysis on the data. Common trading strategies include momentum, mean reversal, and more. We would also like to be able to create a visualiation
dashboard to explore the data. We then would include this on our ANOVA github page.
We used the cryptocurrency index Crix as the market: http://crix.hu-berlin.de/
## Requirements

Python3 is required. All packages used are in the requirements.txt file.

### Getting Started

Most likely to use plotly to construct the dashboard: https://plot.ly/

Examples of good dashboards: https://plot.ly/dash/gallery/uber-rides/

First let's get this working on our own Jupyter notebooks.

Good tutorial to learn git: https://www.atlassian.com/git/tutorials

Tutorial used to build with Flask is:http://adilmoujahid.com/posts/2015/01/interactive-data-visualization-d3-dc-python-mongodb/

Code for Flask is inside the project directory.

## Metrics used

### Bollinger bands
Bollinger bands are simply two lines that are (usually) 2 standard deivations above and below the moving average measure. The bollinger band is a measure of volatility, so the wider the bands, the more volatile the prices are, and vice versa. In the ANOVA project code, simply toggle the standard deviation argument and window argument to adjust the dimensions of the measures.

### Ichimoku Kinkō Hyō
ichimoku Kinkō Hyō translates literally to 'one glance equilibrium chart' and is based on the moving average. Ichimoku, as well as the moving average, are simply ways of identifying trends in prices, which we can see if we toggle the window length of the measures. Ichimoku and the simple moving average are both hampered by extrema, such as we see in bitcoin. For this reason, Ichimoku is more suited to short time periods. NOTE that the Ichimoku actually contains a set of lines, however the 'peripheral' lines are omitted as Bollinger bands are far superior measures of price volatility and trend.

### Simple Moving Average
The simple moving average is computed by taking the arithmetic mean of data points between specified start and end dates, however, these dates can be moved, which changes the sum on the numerator and 'N' in the denominator. Simple moving averages can act as the 'support' for prices, whereby the prices tend to stay above the SMA, or as the 'resistance', where the SMA is the upper limit for the prices. In both cases, the SMA is a way for a trader to know when to sell, by spotting subtleties about the SMA curvature.


_General disclaimer: All of the above measures are entirely subjective and are subject to their inputs. This can be clearly seen by calculating the SMA for different window sizes._

## Built With

* BeautifulSoup

* Care and Coffee

## Authors

The ANOVA Project - Chris Hyland, Eileen Wang, Alex Oh.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgements

* https://realpython.com/blog/python/web-development-with-flask-fetching-data-with-requests/ for tutorial
* https://www.kaggle.com/jackml/cryptocurrency-historical-data for historial data.
* Lots of coffee!
