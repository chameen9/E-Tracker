from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

train_data_gen = ImageDataGenerator(rescale=1./255)
val_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
    'data/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)
val_generator = val_data_gen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

emotional_model = Sequential()

emotional_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotional_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotional_model.add(MaxPooling2D(pool_size=(2, 2)))
emotional_model.add(Dropout(0.25))

emotional_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotional_model.add(MaxPooling2D(pool_size=(2, 2)))
emotional_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotional_model.add(MaxPooling2D(pool_size=(2, 2)))
emotional_model.add(Dropout(0.25))

emotional_model.add(Flatten())
emotional_model.add(Dense(1024, activation='relu'))
emotional_model.add(Dropout(0.5))
emotional_model.add(Dense(7, activation='softmax'))

emotional_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])


emotional_model_info = emotional_model.fit_generator(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=50,
    validation_data=val_generator,
    validation_steps=7178 // 64
)

model_json = emotional_model.to_json()
with open('Model/emotional_model.json', "w") as json_file:
    json_file.write(model_json)

emotional_model.save_weights('emotional_model.h5')
# accuracy = 0.86
