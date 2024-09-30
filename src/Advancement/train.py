import os
from preprocess import create_generators
from model import build_cnn_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image
import os

# Get the current directory of train.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up to the base directory (medical-image-classification)
base_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Construct the path to val_data
train_dir = os.path.join(base_dir, 'data','chest_xray', 'train')
val_dir = os.path.join(base_dir, 'data','chest_xray', 'val')
test_dir = os.path.join(base_dir, 'data','chest_xray', 'test')

model_path = os.path.join(base_dir, 'saved_models', 'advance_model.keras')
model_path_version2 = os.path.join(base_dir, 'saved_models', 'model_cnn_v2.keras')
history_path = os.path.join(base_dir, 'saved_models', 'advancement_history.pkl')

# Generate data
train_generator, val_generator, test_generator = create_generators(train_dir, val_dir, test_dir)

# Build the model
model = build_cnn_model()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# Save the model
model.save(model_path_version2)

# Save the training history
import pickle
with open(history_path, 'wb') as file:
    pickle.dump(history.history, file)