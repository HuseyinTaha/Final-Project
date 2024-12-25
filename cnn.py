import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns


def create_model(input_shape):
    model = tf.keras.Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    return model


def prepare_data(train_dir):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.1
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        color_mode='rgb',
        batch_size=64,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        color_mode='rgb',
        batch_size=64,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator



def train_model(model, train_generator, val_generator):
    optimizer = SGD(learning_rate=0.005, momentum=0.95, decay=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # steps_per_epoch = len(train_generator) // train_generator.batch_size
    # validation_steps = len(val_generator) // val_generator.batch_size

    history = model.fit(
        train_generator,
        # steps_per_epoch=steps_per_epoch,
        epochs=200,
        validation_data=val_generator,
        # validation_steps=validation_steps,
        callbacks=[early_stopping]
    )

    return history


def plot_history(history):
    plt.plot(history.history['loss'],label='Eğitim')
    epochs = range(0, len(history.history['val_loss']), 5)
    val_loss_points = [history.history['val_loss'][i] for i in epochs]
    plt.plot(epochs, val_loss_points, 'o-',label='Doğrulama')
    plt.title('Model Kaybı')
    plt.ylabel('Kayıp')
    plt.xlabel('Epok')
    plt.legend(['Eğitim', 'Doğrulama'], loc='upper right')
    plt.show()

    plt.plot(history.history['accuracy'],label='Eğitim')
    epochs = range(0, len(history.history['val_accuracy']), 5)
    val_acc_points = [history.history['val_accuracy'][i] for i in epochs]
    plt.plot(epochs, val_acc_points, 'o-', label='Doğrulama')
    plt.title('Model Doğruluğu')
    plt.ylabel('Doğruluk')
    plt.xlabel('Epok')
    plt.legend(['Eğitim', 'Doğrulama'], loc='upper left')
    plt.show()

def plot_classification_report(model, val_generator):
    val_generator.reset()
    y_true = val_generator.classes

    y_pred = model.predict(val_generator, steps=len(val_generator),verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    labels = list(val_generator.class_indices.keys())

    report = classification_report(y_true, y_pred_classes, target_names=labels)
    print("Classification Report:\n",report)


def plot_confusion_matrix(model, val_generator):

    val_generator.reset()
    y_true = val_generator.classes

    y_pred = model.predict(val_generator,steps=len(val_generator), verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percentage = np.round(cm_normalized * 100, 1)

    labels = []
    for i in range(len(cm)):
        row = []
        for j in range(len(cm[i])):
            row.append(f"{cm[i, j]}\n({cm_percentage[i, j]}%)")
        labels.append(row)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=labels, fmt='', cmap='RdYlGn', cbar=True, xticklabels=val_generator.class_indices,
                yticklabels=val_generator.class_indices)

    plt.xlabel('Tahmin')
    plt.ylabel('Doğruluk')
    plt.title('Konfüzyon Matrisi')
    plt.show()


def main():

    train_dir = "train_resized"
    model = create_model(input_shape=(48, 48, 3))
    model.summary()

    train_generator, val_generator = prepare_data(train_dir)
    history = train_model(model, train_generator, val_generator)

    plot_history(history)

    print("\nValidation Set Report:")
    plot_classification_report(model, val_generator)
    plot_confusion_matrix(model, val_generator)

    model.save('my_affectNet_model.h5')


if __name__ == "__main__":
    main()