import csv

def simulation(parameter,test=None):
    if test == None:
        #Please call your simulation program with the input parameter 
        #and return its success rate.
        print('Simulation')



    elif test == 'Newtonian':
        with open('example/simulation_example(Newtonian).csv', 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                #print([float(v) for v in row[:-1]], parameter)
                if parameter == [float(v) for v in row[:-1]]:
                    return  float(row[-1])
            
    elif test == 'Langevin':
        with open('example/simulation_example(Langevin).csv', 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                #print([float(v) for v in row[:-1]], parameter)
                if parameter == [float(v) for v in row[:-1]]:
                    return  float(row[-1])
