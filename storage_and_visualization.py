# Save processed spectrogram features and labels
# Stores dataset in NumPy format for efficient loading during model training.
np.save("X.npy", X)
np.save("y.npy", y)

#Visual sanity check
# Displays a sample Mel-spectrogram to verify feature correctness.
import matplotlib.pyplot as plt

plt.imshow(X[0], cmap="magma")
plt.title(f"Label: {y[0]}")
plt.colorbar()
plt.show()

#Class distribution analysis
# Verifies that all instrument classes have equal representation.
labels_unique, counts = np.unique(y, return_counts=True)

plt.bar(labels_unique, counts)
plt.xticks(labels_unique, label_map.keys())
plt.xlabel("Instrument")
plt.ylabel("Samples")
plt.title("Class Distribution")
plt.show()
