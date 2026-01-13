#Class distribution analysis
# Verifies that all instrument classes have equal representation.
labels_unique, counts = np.unique(y, return_counts=True)

plt.bar(labels_unique, counts)
plt.xticks(labels_unique, label_map.keys())
plt.xlabel("Instrument")
plt.ylabel("Samples")
plt.title("Class Distribution")
plt.show()
