import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocess import create_generators



# Get the current directory of train.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up to the base directory (medical-image-classification)
base_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Construct the path to val_data
train_dir = os.path.join(base_dir, 'data','chest_xray', 'train')
val_dir = os.path.join(base_dir, 'data','chest_xray', 'val')
test_dir = os.path.join(base_dir, 'data','chest_xray', 'test')

model_dir = os.path.join(base_dir, 'saved_models', 'model_ccn_v1.keras')
history_dir = os.path.join(base_dir, 'saved_models', 'history.pkl')
fig_dir = os.path.join(base_dir, 'saved_models', 'training_validation_plots_v1.png')

# Load the model
model = load_model(model_dir)

# Generate data
_, _, test_generator = create_generators(train_dir, val_dir, test_dir)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Load training history (optional if saved during training)
import pickle
with open(history_dir, 'rb') as file:
    history = pickle.load(file)

# Plot training and validation accuracy and loss
acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig(fig_dir)

plt.show()