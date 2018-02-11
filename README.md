# Cryptocurrency Technical Analysis
This project seeks to scrape Cryptocurrency data from [coinmarketcap](https://coinmarketcap.com/) and into a CSV file. Then, we will run some technical analysis on the data. Common trading strategies include momentum, mean reversal, and more. We would also like to be able to create a visualiation
dashboard to explore the data. We then would include this on our ANOVA github page.
We used the cryptocurrency index Crix as the market: http://crix.hu-berlin.de/

We have build a simple dashboard showing the metrics after running a simple linear regression between cryptocurrencies as seen below:


![Linear Regression]({{ "/Images/Simple-Analysis.png" | absolute_url }})


We have also built an Ichimoku plot alongisde candle stick charts:


![Ichimoku]({{ "/Images/Ichimoku.png" | absolute_url }})



## Requirements

Python3 is required. All packages used are in the requirements.txt file.

If you are using Mac OSX and a virtualenv, the files for Bokeh plots will not run due to Matplotlib's backend not playing nicely. Instead, we need to use venv, which is same thing as a virtualenv except it only works for Python3, which is fine since that's the only language we support. https://docs.python.org/3/library/venv.html
From that, just run venv as you would normally would for the virtual environment.

To run the Bokeh code on their servers, just go: bokeh serve --show Tech-analysis.plots.py
### Getting Started

To run this dashboard, simply clone this repo. Afterwards, after ensuring you all the required packages, then you can simply run:

```bash
bokeh serve --show Tech-analysis-plots.py
```

Voila! Your beautiful crypto dashboard should be up and running.


Heroku:

Account: admin@anovaprojectusyd.org
password: Anovaproject2003
### PostgreSQL Local Setup

Installation: The easiest way to install is either by __homebrew__ or http://postgresapp.com/ (for Mac Users) Note that you can install any version you want since we will use Heroku in the end.

Command Line Tools: Both options above come with the command line tools. Just be sure to include the bin in your PATH.

Creating your local db:
```
mkdir /usr/local/var/postgres
initdb
```

Starting your server manually: `pg_ctl -D /usr/local/var/postgres start`

Now you can connect to the database (default one is postgres): `psql postgres` (a console should start).

Create a new database through the console: `CREATE DATABASE <name> ;` (I named it _anova-crypto_.) Alternatively, you could use `createdb <name>` in your command line.

From this point, you can play around now. For more references, you can include `--help` option in the command for more information.

### Flask + SQLAlchemy + PostgreSQL

Install __autoenv__ or similar tools if haven't already: This is particularly useful for development since we often deal with enviroments. They can automatically switch your virtual environment and define environment variables.

Run Flask App via `flask run`. However, we need to set the `FLASK_APP` environment variable to the flask app file which is `app.py` in this case.

We can create models (or tables) in `models.py`. Now we can actually create tables in the connected database from these predefined models. Note that the database you want to connect to can be set through `DATABASE_URL` environment variable but since I'm testing on my local machine, it's `postgresql://localhost/anova-crypto`.

__Migration Repository__: This is a directory that stores scripts that are created whenever the database schema is changed. These scripts record the changes and can be applied to your database. This is useful when the database structure needs to be changed.
- `flask db init`: Create the migration repository (already did here).
- `flask db migrate`: Automatically generate the migration scripts.
- `flask db upgrade`: Apply the changes
- `flask db downgrade`: Undo the last migration.

Playing with SQLAlchemy: The documentation is really useful and clear http://flask-sqlalchemy.pocoo.org/2.3/. Also, we can use `flask shell` which is similar to python shell but we can pre-import objects from the app file.


## Metrics used

### Bollinger bands
Bollinger bands are simply two lines that are (usually) 2 standard deivations above and below the moving average measure. The bollinger band is a measure of volatility, so the wider the bands, the more volatile the prices are, and vice versa. In the ANOVA project code, simply toggle the standard deviation argument and window argument to adjust the dimensions of the measures.

### Ichimoku Kinkō Hyō
ichimoku Kinkō Hyō translates literally to 'one glance equilibrium chart' and is based on the moving average. Ichimoku, as well as the moving average, are simply ways of identifying trends in prices, which we can see if we toggle the window length of the measures. Ichimoku and the simple moving average are both hampered by extrema, such as we see in bitcoin. For this reason, Ichimoku is more suited to short time periods. NOTE that the Ichimoku actually contains a set of lines, however the 'peripheral' lines are omitted as Bollinger bands are far superior measures of price volatility and trend. More information can be found at: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud

### Simple Moving Average
The simple moving average is computed by taking the arithmetic mean of data points between specified start and end dates, however, these dates can be moved, which changes the sum on the numerator and 'N' in the denominator. Simple moving averages can act as the 'support' for prices, whereby the prices tend to stay above the SMA, or as the 'resistance', where the SMA is the upper limit for the prices. In both cases, the SMA is a way for a trader to know when to sell, by spotting subtleties about the SMA curvature.


_General disclaimer: All of the above measures are entirely subjective and are subject to their inputs. This can be clearly seen by calculating the SMA for different window sizes._

## Built With

* BeautifulSoup

* Care and Coffee

## Authors

The ANOVA Project - Eileen Wang, Supavit Dumrongprechachan, Chris Hyland, Alex Oh.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgements

* https://realpython.com/blog/python/web-development-with-flask-fetching-data-with-requests/ for tutorial
* https://www.kaggle.com/jackml/cryptocurrency-historical-data for historial data.
* Lots of coffee!
