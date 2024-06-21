import csv

def read(file_name, dim, nb_max_indiv):
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile)

        if (dim == 1):
            
            i = 0
            list_id = []
            data_t = []
            data_y = []

            for row in reader:
                if (i != 0):
                    identifiant = int(row[0])
                    time = float(row[1])
                    value = float(row[2])

                    k = -1
                    j = 0
                    for elem in list_id:
                        if (identifiant == elem):
                            k = j
                        j = j + 1

                    if (k == -1):
                        if (len(list_id) < nb_max_indiv):
                            list_id.append(identifiant)
                            data_t.append([time])
                            data_y.append([value])
                    else:
                        data_t[k].append(time)
                        data_y[k].append(value)
                
                i = i + 1
        else:
            
            i = 0
            list_id = []
            data_t = []
            data_y = []

            for row in reader:
                if (i != 0):
                    identifiant = int(row[0])
                    time = float(row[1])
                    value = [float(row[l + 2]) for l in range(dim)]

                    k = -1
                    j = 0
                    for elem in list_id:
                        if (identifiant == elem):
                            k = j
                        j = j + 1

                    if (k == -1):
                        if (len(list_id) < nb_max_indiv):
                            list_id.append(identifiant)
                            data_t.append([time])
                            data_y.append([value])
                    else:
                        data_t[k].append(time)
                        data_y[k].append(value)
                
                i = i + 1
        return data_t, data_y

data_t, data_y = read("test.csv", 1, 20)
print(data_t)
print(data_y)
