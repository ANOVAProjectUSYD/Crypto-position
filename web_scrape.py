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
    # Iterate through table entries by row.
    for row in table.find_all('tr'):
        cells = row.find_all("td")
        # Gets ranking.
        rn = cells[0].get_text()
        # Gets name of currency.
        rn2 = cells[1].get_text()
        # Gets market cap
        rn3 = cells[2].get_text()
        # Gets price
        rn4 = cells[3].get_text()
        # Gets change in 24 hours.
        rn5 = cells[6].get_text()
        print(rn2)
    return


def main():
    parse_site(get_website())


if __name__ == "__main__":
    main()
