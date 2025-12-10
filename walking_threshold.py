import matplotlib.pyplot as plt

# Data points
#distance = [0, 400, 1800,2400] walkscore = [100, 95, 10, 0]
distance = [0, 400, 800, 1200, 1600, 2000]
walkscore = [100, 95, 70, 40 , 10, 0]


# Plot
plt.figure(figsize=(6, 4))
plt.plot(distance, walkscore, color='steelblue', linewidth=1.5)

# Dotted helper lines
plt.hlines(95, 0, 400, colors='gray', linestyles='dotted')
plt.vlines(400, 0, 95, colors='gray', linestyles='dotted')

plt.hlines(10, 0, 1600, colors='gray', linestyles='dotted')
plt.vlines(1600, 0, 10, colors='gray', linestyles='dotted')

# Axis labels
plt.xlabel("Distance (m)")
plt.ylabel("WalkScore")

# Axis limits
plt.xlim(0, 2500)
plt.ylim(0, 100)

# Ticks to display the key values
plt.xticks([0, 400, 800, 1200, 1600, 2000])
plt.yticks([0, 10, 40, 70, 95, 100])

# Clean up the borders
plt.box(True)


plt.tight_layout()
plt.show()
