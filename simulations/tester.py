import time

# Define the final values
final_value1 = 100
final_value2 = 200
final_value3 = 300

# Define the initial values
initial_value1 = 0
initial_value2 = 0
initial_value3 = 0

# Define the rate of change (increment) per iteration
rate_of_change1 = 5
rate_of_change2 = 10
rate_of_change3 = 15

# Define the number of iterations
num_iterations = 10

# Calculate the increment per iteration
increment1 = (final_value1 - initial_value1) / num_iterations
increment2 = (final_value2 - initial_value2) / num_iterations
increment3 = (final_value3 - initial_value3) / num_iterations

# Loop through the iterations
for i in range(num_iterations + 1):
    # Update the values
    value1 = initial_value1 + (increment1 * i)
    value2 = initial_value2 + (increment2 * i)
    value3 = initial_value3 + (increment3 * i)

    # Print the current values
    print(f"Value 1: {value1}, Value 2: {value2}, Value 3: {value3}")

    # Pause for a short duration (e.g., 1 second)
    time.sleep(1)
