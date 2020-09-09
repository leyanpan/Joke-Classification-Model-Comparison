import csv


def write_negative_data():
    with open('puns_pos_neg_data.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == '1':
                with open('positive_data_file.csv', 'a') as csv_pos:
                    writer = csv.writer(csv_pos)
                    writer.writerow(row)
            if row[0] == '-1':
                with open('negative_data_file.csv', 'a') as csv_neg:
                    writer = csv.writer(csv_neg)
                    writer.writerow(row)

