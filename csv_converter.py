import csv

def openFile(name):
    '''Open file and rewrite as csv file.'''
    if '.csv' in name:
        name = name.split('.csv')[0]
    file_name = name + '.csv'
    try:
        with open(file_name, 'r') as read_file:
            new_file_name = name + '_final.csv'
            temp_file = open(new_file_name, 'w')
            writer = csv.writer(temp_file)
            reader = csv.reader(read_file)
            for row in reader:
                line = row[0].split(';')
                writer.writerow([line[0], line[1],
                                 line[2], line[3],
                                 line[4], line[5],
                                 line[6]])
            temp_file.close()
            read_file.close()
    except Exception as e:
        print("Error occured")
        print(e)

def main():
    file_name = input("Enter csv file name: ")
    openFile(file_name)

if __name__ == "__main__":
    main()
