import numpy as np
import matplotlib.pyplot as plt

def load_data(fname):

    with open(fname) as f:
        content = f.readlines()
    print (len(content))        

    data  = []
    for i in content:
        data.append(float(i))
    print (len(data))        

    x = np.asarray(data)

    # the histogram of the data
    n, bins, patches = plt.hist(x, 1000,  facecolor='g', alpha=0.75)

    plt.xlabel('Energy')
    plt.ylabel('Probability')
    plt.title('Histogram of Energy')
    plt.grid(True)
    plt.show()


load_data('./exel2')
