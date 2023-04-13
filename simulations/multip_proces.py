import matplotlib.pyplot as plt
from multiprocessing import Process
for x in range(6,8):
    print(x)


x_vals = [0,1,2,3,4,5]
y_vals = [1,2,6,2,1,3]
def create_plot():
    fig, ax = plt.subplots()
    # add plot code here
    plt.plot(x_vals,y_vals)
    plt.show()


def for_loop():
    for k in range(3):
        print(k)

if __name__ == '__main__':
    p = Process(target=create_plot)
    p.start()
    p1 = Process(target=for_loop)
    p1.start()
    p.join()
    p1.join()