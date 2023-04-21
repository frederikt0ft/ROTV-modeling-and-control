import matplotlib.pyplot as plt

# create some sample data
x = [1, 2, 3]
y1 = [4, 5, 6]
y2 = [7, 8, 9]

# create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2)

# plot data on the subplots
ax1.plot(x, y1)
ax2.plot(x, y2)

# adjust the position of the subplots
plt.subplots_adjust(left=0.3, right=0.7, bottom=0.1, top=0.9, wspace=0.3)

# show the plot
plt.show()