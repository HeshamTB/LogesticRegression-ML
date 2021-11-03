
def read_csv_file(filename: str, skip_first: bool) -> list:
    import csv
    data = list()
    with open(filename, 'r') as csvFile:
        csvReader = csv.reader(csvFile)
        for i, row in enumerate(csvReader):
            if i == 0:
                if skip_first:
                    continue
                else:
                    data.append(row)  # pass first row (feat names)
            else:
                data.append(row)
    return data


def print_feature_indcies(data) -> None:
    for i, val in enumerate(data[0]):
        print((i, val))


def selected_columns(data: list, columns: list) -> None:
    columns = set(columns) # Removes duplicates.
    for rowIndex in range(len(data)):
        newRow = list()
        for featureIndex in range(len(data[rowIndex])):
            if featureIndex in columns:
                try:
                    newRow.append(float(data[rowIndex][featureIndex]))
                except ValueError:
                    newRow.append(0.0)
        # Replace new row
        data[rowIndex] = newRow

def featurescaling(array):
    #  feature scaling
    biggest_value = 0
    counter_list = 0
    counter_element = 0
    for mnm in range(len(array[0])):
        for me in array:
            if (int(array[counter_list][counter_element])) > biggest_value:
                biggest_value = (int(array[counter_list][counter_element]))
            counter_list = counter_list + 1
        counter_list = 0

        for mee in array:
            array[counter_list][counter_element] = ((int(array[counter_list][counter_element])) / biggest_value)
            counter_list = counter_list + 1

        counter_list = 0
        biggest_value = 0
        counter_element = counter_element + 1
    return array