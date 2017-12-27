from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.error import HTTPError


def get_website():
    '''Ensure call to website is successful.'''
    try:
        html = urlopen('https://coinmarketcap.com/')
    except HTTPError as e:
        # Page not found exception.
        print('Cannot connect to website at this time.')
        return
    if html is None:
        # Server is not found.
        print('Server not found.')
        return
    else:
        return html


def parse_site(html):
    '''Parses the site and returns a list of cryptocurrency objects.'''
    bs_Obj = BeautifulSoup(html, "html.parser")
    # Table in which currencies reside.
    table = bs_Obj.find("table", {"id": "currencies"}).find('tbody')
    currency_list = []
    # Iterate through table entries by row.
    for row in table.find_all('tr'):
        cells = row.find_all("td")
        # Create crypto object.
        crypto_obj = create_crypto(cells)
        currency_list.append(crypto_obj)
    return currency_list


class Currency:
    def __init__(self, name, rank, market_cap, price, price_change):
        self.name = name
        self.rank = rank
        self.market_cap = market_cap
        self.price = price
        self.price_change = price_change


def create_crypto(cells):
    '''Create list of currency objects.'''
    rank = str(cells[0].get_text())
    name = str(cells[1].get_text())
    market_cap = str(cells[2].get_text())
    price = str(cells[3].get_text())
    price_change = str(cells[6].get_text())
    return Currency(name, rank, market_cap, price, price_change)


def main():
    listOfStuff = parse_site(get_website())
    # Whole list.
    # print(listOfStuff)
    # # Access first object in list.
    # print(listOfStuff[0])
    # # Access the name attribute in first object.
    # print(listOfStuff[0].name)
    for x in listOfStuff:
        print(x.name)
        print(x.price)
        print(x.rank)


if __name__ == "__main__":
    main()
