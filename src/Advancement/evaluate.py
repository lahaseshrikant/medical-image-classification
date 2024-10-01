import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocess import create_generators
import os
import pickle

# Get the current directory of train.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up to the base directory (medical-image-classification)
base_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Paths for data and model
train_dir = os.path.join(base_dir, 'data', 'chest_xray','train')
val_dir = os.path.join(base_dir, 'data', 'chest_xray','val')
test_dir = os.path.join(base_dir, 'data', 'chest_xray','test')

model_dir = os.path.join(base_dir, 'saved_models', 'advance_model.keras')
history_dir = os.path.join(base_dir, 'saved_models', 'advancement_history.pkl')
fig_dir = os.path.join(base_dir, 'saved_models', 'training_validation_plots_advanced.png')

# Load the best model
model = load_model(model_dir)

# Generate data
_, val_generator, test_generator = create_generators(train_dir, val_dir, test_dir)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Load training history (if saved during training)
with open(history_dir, 'rb') as file:
    history = pickle.load(file)

# Extract history data
acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']

# Ensure the number of epochs matches for all the arrays
epochs_range = range(len(acc))

# Print shapes to debug any mismatch
print(f"Training Accuracy History Length: {len(acc)}")
print(f"Validation Accuracy History Length: {len(val_acc)}")
print(f"Training Loss History Length: {len(loss)}")
print(f"Validation Loss History Length: {len(val_loss)}")
print(f"Epochs Range: {len(epochs_range)}")

# Plot training and validation accuracy and loss
plt.figure(figsize=(14, 6))

# Plot training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', color='b', linestyle='--', linewidth=2)
plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='g', linewidth=2)
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', color='r', linestyle='--', linewidth=2)
plt.plot(epochs_range, val_loss, label='Validation Loss', color='m', linewidth=2)
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)

# Save the plot to a file
plt.savefig(fig_dir)

# Show the plot
plt.show()

# Print final metrics
print(f"Final Training Accuracy: {acc[-1] * 100:.2f}%")
print(f"Final Validation Accuracy: {val_acc[-1] * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
