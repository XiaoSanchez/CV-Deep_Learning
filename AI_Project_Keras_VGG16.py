from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

from keras.models import load_model

num_classes = 2
batch_size = 100
image_size = (224, 224)

generator = ImageDataGenerator(preprocessing_function=preprocess_input)

training_generator = generator.flow_from_directory(
    "concrete_data_week4/train",
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
)
validation_generator = generator.flow_from_directory(
    "concrete_data_week4/valid",
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
)
model_vgg16 = Sequential()

model_vgg16.add(VGG16(include_top=False, pooling="avg", weights="imagenet"))
model_vgg16.add(Dense(num_classes, activation="softmax"))

model_vgg16.layers[0].trainable = False

model_vgg16.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

num_epochs = 2
steps_per_epoch_training = len(training_generator)
steps_per_epoch_validation = len(validation_generator)

history_vgg16 = model_vgg16.fit_generator(
    training_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


num_classes = 2
image_resize = 224
batch_size_training = 100
batch_size_validation = 100

data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

train_generator = data_generator.flow_from_directory(
    'concrete_data_week4/train',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
    'concrete_data_week4/valid',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical')

model = Sequential()

model.add(ResNet50(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))

model.add(Dense(num_classes, activation='softmax'))
model.layers[0].trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

steps_per_epoch_training = len(train_generator)
steps_per_epoch_validation = len(validation_generator)
num_epochs = 2

history_resnet = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
)
model.save('resnet_model.h5')

model_resnet50 = load_model("resnet_model.h5")

testing_generator = generator.flow_from_directory(
    "concrete_data_week4/test", target_size=image_size, shuffle=False,
)
## for the VGG16
performance_vgg16 = model_vgg16.evaluate_generator(testing_generator)
print("Performance of the VGG16-trained model")
print("Loss: {}".format(round(performance_vgg16[0], 5)))
print("Accuracy: {}".format(round(performance_vgg16[1], 5)))

## for the RESNET
performance_resnet50 = model_resnet50.evaluate_generator(testing_generator)
print("Performance of the ResNet50-trained model")
print("Loss: {}".format(round(performance_resnet50[0], 5)))
print("Accuracy: {}".format(round(performance_resnet50[1], 5)))

predictions_vgg16 = model_vgg16.predict_generator(testing_generator, steps=1)


def print_prediction(prediction):
    if prediction[0] > prediction[1]:
        print("Negative ({}% certainty)".format(round(prediction[0] * 100, 1)))
    elif prediction[1] > prediction[0]:
        print("Positive ({}% certainty)".format(round(prediction[1] * 100, 1)))
    else:
        print("Unsure (prediction split 50/50)")


print("First five predictions for the VGG16-trained model:")
for i in range(5):
    print_prediction(predictions_vgg16[i])

    
predictions_resnet50 = model_resnet50.predict_generator(testing_generator, steps=1)
print("First five predictions for the ResNet50-trained model:")
for i in range(5):
    print_prediction(predictions_resnet50[i])