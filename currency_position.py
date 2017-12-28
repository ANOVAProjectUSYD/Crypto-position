from web_scrape.py import parse_site, get_website
import csv
import shutil
from tempfile import NamedTemporaryFile

# 2 csv files created. Latest info and log of previous ones.
# Log of previous ones shows spreadsheet with each row being a cryptocurrency
# and their movements. So each sheet will track a different attribute about
# the cryptocurrency.

# Purpose: Produce spreadsheet highlighting changes in position over 24 hours.
# Time used is 9:00am each day when we reset the position.


def read_data(objects):
    '''Takes in list of currency objects and converts it into list of dictionaries.'''
    list_objects = []
    for obj in objects:
        # TODO: Convert list of objects into list of dictionaries for each obj.


def update_log(objects):
    '''Update our log of cryptocurrencies and add new currencies if needed.'''
    # When updating our log file, it is best to do it via a temporary file.
    # More info: https://stackoverflow.com/questions/16020858/inline-csv-file-editing-with-python/16020923#16020923
    temp_file = NamedTemporaryFile(mode='w', delete=False)
    file_name = 'currency_log.csv'
    # TODO: Read in the data from the read_data function.
    with open(file_name, r) as csv_file, temp_file:
        reader = csv.DictReader(csv_file, fieldnames=fields)
        writer = csv.DictWriter(temp_file, fieldnames=fields)
        for row in reader:
            row = {'Name': row}
            writer.writerow(row)
            # TODO: Write out to the file and update log.
    # Transfer content over.
    shutil.move(tempfile.name, filename)
