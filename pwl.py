import matplotlib.pyplot as plt



# Data points (piecewise linear breakpoints)
distance = [0, 400, 1800, 2400, 3000]
walkscore = [100, 95, 10, 0, 0]


plt.grid(False)
plt.xticks([0, 400, 1800, 2400])
plt.yticks([0, 10, 95, 100])

plt.figure(figsize=(6,4))

# Plot the main line
plt.plot(distance, walkscore, marker='o')

# Add dashed guide lines (optional, like in your figure)
plt.axvline(x=400, linestyle='--')
plt.axvline(x=1800, linestyle='--')
plt.axhline(y=95, linestyle='--')
plt.axhline(y=10, linestyle='--')

# Labels
plt.xlabel("Distance (m)")
plt.ylabel("WalkScore")

# Axis limits
plt.xlim(0, 2500)
plt.ylim(0, 100)

plt.tight_layout()
plt.show()