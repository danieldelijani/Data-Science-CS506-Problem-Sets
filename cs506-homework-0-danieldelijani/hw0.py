# HW0 Daniel Delijani

import math
import copy
import random

# 1) Parse and Preprocess Data 

def import_data(filename):
    # reading from the file, setting up an initial dataset array
    file = open(filename, "r")
    data = file.read()
    patients = data.split("\n")
    for i in range(len(patients)):
        patients[i] = patients[i].split(",")
    for r in range(len(patients)):
        for c in range(len(patients[0])):
            if patients[r][c] == '?':
                patients[r][c] = float("NaN")
            else:
                patients[r][c] = float(patients[r][c])
    #forming and returning x, y
    X = []
    y = []
    for r in range(len(patients)):
        X += [patients[r][0:-1]]
        y += [patients[r][-1]]

    file.close()
    return X, y

# 2) Impute or delete missing entries -----------------------------------------------------------------------------------------------------------------

# Part a

# iterate from column to column
# compute the median of that column
# iterate down the column, and if find a NaN, replace it w median of that column and keep going

def createarray(column, lst):
    # takes 2-d array, and when you give it a column it creates an array with all the values from that column
    arr = []
    for r in range(len(lst)):
        arr += [lst[r][column]]
    return arr

def median(lst):
    # takes 1-d array and returns the median, ignoring NaN values
    cleanlist = []
    for i in range(len(lst)):
        if not math.isnan(lst[i]):
            cleanlist += [lst[i]]
    cleanlist.sort()

    length = len(cleanlist)

    if length % 2 == 1:
        return cleanlist[length//2]
    else:
        x1 = cleanlist[length//2]
        x2 = cleanlist[(length//2) -1]
        return (x1 + x2)/2.0
    
def impute_missing(x):
    newarray = copy.deepcopy(x)
    for c in range(len(newarray[0])):
        med = median(createarray(c, newarray))
        for r in range(len(newarray)):
            if math.isnan(newarray[r][c]):
                newarray[r][c] = med
    return newarray


# Part B

"""the main advantage of using a median rather than a mean, is that the median
   value will not be so largely affected by outliers in the data set, allowing
   us to gain a better estimate for a value in which most of the dataset is near"""

# Part C
def discard_missing(x, y):
    newarray = copy.deepcopy(x)
    newy = copy.deepcopy(y)
    r = 0
    for ignoredvar in range(len(newarray)): # using my variable as row iterator so I can manipulate as I please
        for c in range(len(newarray[0])):
            if math.isnan(newarray[r][c]):
                newarray.pop(r)
                newy.pop(r)
                r -= 1 # to adjust for one less row
                break
        r += 1
    return newarray, newy
    

# 3) Working with the data -----------------------------------------------------------------------------------------------------------------

# Part A

def shuffle_data(x, y):
    newx = copy.deepcopy(x)
    newy = copy.deepcopy(y)
    
    lst = [i for i in range(len(x))]
    random.shuffle(lst)

    for i in range(len(newx)):
        newx[i] = copy.deepcopy(x[lst[i]])
        newy[i] = y[lst[i]]
    return newx, newy


# Part B

def computemean(lst):
    """Takes 1-d array and returns the mean"""
    sum = 0.0
    for i in range(len(lst)):
        sum += lst[i]
    return sum / len(lst)

def standarddev(lst):
    """ takes 1-d array and computes standard devation """
    mean = computemean(lst)
    sum = 0.0
    for i in range(len(lst)):
        diff = lst[i] - mean
        diffsquared = diff ** 2
        sum += diffsquared
    std = sum / (len(lst) - 1.0)
    std = std ** .5
    return std

def compute_std(x):
    arr = copy.deepcopy(x)
    std = []
    for i in range(len(arr[0])):
        coldata = createarray(i, arr)
        stdcol = standarddev(coldata)
        std += [stdcol]
    return std


# Part C

def remove_outlier(x, y):
    newarray = copy.deepcopy(x)
    standevs = compute_std(x)
    newy = copy.deepcopy(y)

    r = 0
    for ignoredvar in range(len(newarray)): # using my variable as row iterator so I can manipulate as I please
        for c in range(len(newarray[0])):
            stdevcol = standevs[c]
            meancol = computemean(createarray(c, newarray))
            distfrommean = newarray[r][c] - meancol
            if stdevcol == 0: # so no div by zero error, because some cols have all the same value stdev is 0
                zscore = 0
            else:
                zscore = distfrommean / stdevcol
            if zscore < -2 or zscore > 2:
                newarray.pop(r)
                newy.pop(r)
                r -= 1 # to adjust for one less row
                break
        r += 1
    return newarray, newy


# Part D 

def standardize_data(x):
    newarray = copy.deepcopy(x)
    standevs = compute_std(x)

    for c in range(len(newarray[0])):
        meancol = computemean(createarray(c, newarray))
        stdevcol = standevs[c]
        for r in range(len(newarray)):
            distfrommean = newarray[r][c] - meancol
            if stdevcol == 0: # so no div by zero error, because some cols have all the same value stdev is 0
                zscore = 0
            else:
                zscore = distfrommean / stdevcol
            newarray[r][c] = zscore
    return newarray

"""
time complexity: calculating standevs and means: O(n), looping through each element O(n) - total: O(n)
space complexity: Have two arrays of size n and some constants -> total: O(n)

 """

# Part 4 - Working with non-numerical data -----------------------------------------------------------------------------------------------------------------

def import_data2(filename):
    # reading from the file, setting up an initial dataset array
    file = open(filename, "r")
    data = file.read()
    patients = data.split("\n")
    patients = patients[1:] # get rid of header
    for i in range(len(patients)):
        patients[i] = patients[i].split(",")
    
    for r in range(len(patients)):
        for c in range(len(patients[0])):
            if c == 0 or c == 1 or c == 2 or c == 6 or c ==7 or c == 8 or c == 10:
                if patients[r][c] == '':
                    patients[r][c] = float('NaN')
                else:
                    patients[r][c] = float(patients[r][c])
            elif c == 5: # gender
                if patients[r][c] == 'female':
                    patients[r][c] = 0.0
                elif patients[r][c] == 'male':
                    patients[r][c] = 1.0
                elif patients[r][c] == '':
                    patients[r][c] == float('NaN')
                else:
                    print(patients[r][c])
                    print('ERROR!!!!!!!!!!')
                    return
            elif c == 12: # cabin
                if patients[r][c] == 'C\r' or patients[r][c] == 'C':
                    patients[r][c] = 0.0
                elif patients[r][c] == 'Q\r' or patients[r][c] == 'Q':
                    patients[r][c] = 1.0
                elif patients[r][c] == 'S\r' or patients[r][c] == 'S':
                    patients[r][c] = 2.0
                elif patients[r][c] == '\r':
                    patients[r][c] == float('NaN')
                else:
                    print(patients[r][c])
                    print('ERROR!!!!!!!!!!')
                    return
    for r in range(len(patients)): # getting rid of unecessary categories
        patients[r] = patients[r][0:3] + patients[r][5:9] + [patients[r][10]] + [patients[r][12]]    

    #forming and returning x, y
    X = []
    y = []
 
    for r in range(len(patients)):
        X += [ [patients[r][0]] + patients[r][2:] ]
        y += [patients[r][1]]

    file.close()
    return X, y


# 5) - Train-test split -----------------------------------------------------------------------------------------------------------------

# Part A

def train_test_split(x, y, t_f):
    newx = copy.deepcopy(x)
    newy = copy.deepcopy(y)

    numtest = int(t_f * len(newx))
    randarr = [i for i in range(len(newx))]
    random.shuffle(randarr)
    testindicies = randarr[0:numtest]
    trainindicies = randarr[numtest:]

    x_train = [copy.deepcopy(x[i]) for i in trainindicies]
    y_train = [y[i] for i in trainindicies]

    x_test = [copy.deepcopy(x[i]) for i in testindicies]
    y_test = [y[i] for i in testindicies]

    return x_train, y_train, x_test, y_test

# Part B

def train_test_CV_split(x, y, t_f, cv_f):
    newx = copy.deepcopy(x)
    newy = copy.deepcopy(y)

    numtest = int(t_f * len(newx))
    numcross = int(cv_f * len(newx))

    randarr = [i for i in range(len(newx))]
    random.shuffle(randarr)
    testindicies = randarr[0:numtest]
    crossindicies = randarr[numtest:numtest+numcross]
    trainindicies = randarr[numtest+numcross:]


    x_train = [copy.deepcopy(x[i]) for i in trainindicies]
    y_train = [y[i] for i in trainindicies]

    x_test = [copy.deepcopy(x[i]) for i in testindicies]
    y_test = [y[i] for i in testindicies]

    x_cv = [copy.deepcopy(x[i]) for i in crossindicies]
    y_cv = [y[i] for i in crossindicies]

    return x_train, y_train, x_test, y_test, x_cv, y_cv