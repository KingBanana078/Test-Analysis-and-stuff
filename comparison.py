def read_temp_csv():
    with open('Temperature.csv') as temp:
        reader = csv.reader(temp)
        return [list(map(float, row)) for row in reader]