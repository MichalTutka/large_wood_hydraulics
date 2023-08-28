import csv

# create an empty list to store the tuples
channel_points = []

# specify the full path to the csv file
file_path = r'D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Python\practice_stations\mc3_stations_channel_practice.csv'

# open the csv file and read the contents
with open(file_path) as csvfile:
    reader = csv.DictReader(csvfile)

    # loop through each row of the csv file
    for row in reader:
        # create a tuple with the values of the current row
        # the first value is assigned to index 0 and the second to index 1
        point = (row['Station'], row['Z'])
        
        # add the tuple to the list
        channel_points.append(point)

print(channel_points)