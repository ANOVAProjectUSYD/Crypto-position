from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib2 import HTTPError


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
    bs_Obj = BeautifulSoup(html)
    my_Divs = bs_Obj.findAll("table", {"id": "currencies"})
    print(my_Divs)
    return


def main():
    get_website()
    parse_site()


if __name__ == "__main__":
    main()
