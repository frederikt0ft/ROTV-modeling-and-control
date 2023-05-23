import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Frame

# Create a Tkinter root window
root = Tk()
root.title("Figure Window Positions")

# Create a frame to hold the figures
frame = Frame(root)
frame.pack(fill='both', expand=True)

# Create the first figure and plot
fig1, ax1 = plt.subplots()
ax1.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])

# Create the second figure and plot
fig2, ax2 = plt.subplots()
ax2.plot([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])

# Create the third figure and plot
fig3, ax3 = plt.subplots()
ax3.plot([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])

# Create FigureCanvasTkAgg widgets to display the figures in the frame
canvas1 = FigureCanvasTkAgg(fig1, master=frame)
canvas1.draw()
canvas1.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)

canvas2 = FigureCanvasTkAgg(fig2, master=frame)
canvas2.draw()
canvas2.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)

canvas3 = FigureCanvasTkAgg(fig3, master=frame)
canvas3.draw()
canvas3.get_tk_widget().grid(row=0, column=2, padx=5, pady=5, columnspan=2)

root.mainloop()
