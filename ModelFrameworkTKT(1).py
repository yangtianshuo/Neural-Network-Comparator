import pickle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import ImageGrab
import tensorflow as tf
import numpy as np
import cv2
matplotlib.use('TkAgg')

# Load and preprocess the data
# the four models we have here are using the same data
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_val = x_val / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)

# Initialize the history, prediction and validation result lists
o_history_list = history_list = []
o_val_loss_list = val_loss_list = []
o_val_acc_list = val_acc_list = []
o_predictions_list = predictions_list = []
o_models = models = []
num_eporchs = 10

def build_model():
    global model1, model2, model3, model4
    model_list = []
    # Build and compile the simple model
    model1 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Build and compile the MLP model
    model2 = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Build and compile the simple CNN model
    model3 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Build and compile the complex CNN model
    model4 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model4.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_list = [model1, model2, model3, model4]
    return model_list

# Create a list of the models, model1...model4 are assigned value
o_models = build_model()

def load_model():
    # If the model and history value pair are already in file, read it from there to avoid training time
    models = []
    history_list = []

    for i in range(4):
        try:
            # Load the model and its training history
            with open(f'model{i + 1}.pkl', 'rb') as f:
                model_dict = pickle.load(f)
            model = model_dict['model']
            models.append(model)
            history = model_dict['history']
            history_list.append(history)
        except:
            # Read the model from built models
            model = o_models[i]

            # Train the model and collect the results
            history = model.fit(x_train, y_train, epochs=num_eporchs, validation_data=(x_val, y_val))

            # Save the model and its training history
            model_dict = {'model': model, 'history': history}
            with open(f'model{i + 1}.pkl', 'wb') as f:
                pickle.dump(model_dict, f)

            models.append(model)
            history_list.append(history)
    return models, history_list


models, history_list = load_model()

# Collect the predictions from the models
predictions_list = []
for model in models:
    predictions = model.predict(x_val)
    predictions_list.append(predictions)

# Collect the validation loss and accuracy from the models
val_loss_list = []
val_acc_list = []
for model in models:
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
o_history_list = history_list
o_predictions_list = predictions_list
o_val_loss_list = val_loss_list
o_val_acc_list = val_acc_list


# Function to predict the digit using the selected models
def predict_digit(image, tmodels):
    # Preprocess the image
    image = image / 255.0
    image = image.reshape(-1, 28, 28, 1)

    # Use the selected models to make predictions
    predictions = []
    for model in tmodels:
        predictions.append(model.predict(image))

    # Return the predictions as a list
    return predictions


# Function to clear the canvas
def clear_canvas(canvas):
    canvas.delete("all")


# Function to submit the drawn image and show the predictions
def submit(tRoot, tcanvas, label, tmodels):
    global selected_model_indices, history_list, predictions_list, \
           val_loss_list, val_acc_list, o_history_list, o_predictions_list, \
           o_val_loss_list, o_val_acc_list
    # Get the image from the canvas
    x = tRoot.winfo_rootx() + tcanvas.winfo_x()
    y = tRoot.winfo_rooty() + tcanvas.winfo_y()
    x1 = x + tcanvas.winfo_width()
    y1 = y + tcanvas.winfo_height()
    image = ImageGrab.grab((x, y, x1, y1))

    # Convert the image to a numpy array
    image = np.array(image)

    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to 28x28 pixels
    image = cv2.resize(image, (28, 28))

    # Display the image
    plt.imshow(image, cmap='gray')

    # Reshape the image to match the input shape of the model
    image = image.reshape(1, 28, 28, 1)
    selected_model_names = [model_name.get() for model_name in ui_model_names if model_name.get() != "Empty"]
    selected_model_indices = []
    for t_model_name in selected_model_names:
        for i, model_name in enumerate(ui_model_names):
            if model_name.get() == t_model_name:
                selected_model_indices.append(i)
                break

    selected_models = [tmodels[i] for i in selected_model_indices]

    history_list = [o_history_list[i] for i in selected_model_indices]
    predictions_list = [o_predictions_list[i] for i in selected_model_indices]
    val_loss_list = [o_val_loss_list[i] for i in selected_model_indices]
    val_acc_list = [o_val_acc_list[i] for i in selected_model_indices]
    #
    # Make predictions using the selected models
    predictions = predict_digit(image, selected_models)

    # Display the predictions
    text = ""
    for i, (prediction, model_name) in enumerate(zip(predictions, selected_model_names)):
        if len(prediction.shape) > 1:
            max_index = np.argmax(prediction, axis=1)
            max_prediction = prediction[np.arange(prediction.shape[0]), max_index]
        else:
            max_index = np.argmax(prediction)
            max_prediction = prediction[max_index]
        text += f"{model_name}: Prediction {max_index} -- {max_prediction}\n"

    label.config(text=text)

    plt.show()

# Initialize the starting and ending points for the line
start_x = None
start_y = None
end_x = None
end_y = None

# Function to draw a line on the canvas
def draw_line(event):
    global start_x, start_y, end_x, end_y
    # Get the current mouse position
    end_x = event.x
    end_y = event.y
    # If this is the first point, just save it as the starting point
    if start_x is None and start_y is None:
        start_x = end_x
        start_y = end_y
    # Otherwise, draw a line from the starting point to the current point
    else:
        canvas.create_line(start_x, start_y, end_x, end_y)
        start_x = end_x
        start_y = end_y

# Function to reset the starting and ending points for the line
def reset_points(event):
    global start_x, start_y, end_x, end_y
    start_x = None
    start_y = None
    end_x = None
    end_y = None
    # canvas.delete("all")

#prediction window to create/show figure, dropdown menu, label result and buttons UI elements by connecting to root object passed
def predictionWindow(tRoot):
    global root, canvas
    tRoot.title("Digit Prediction")
    # Create the canvas for drawing
    tCanvas = tk.Canvas(tRoot, width=280, height=280, bg='white')
    tCanvas.pack()

    # Bind the mouse events to the canvas
    tCanvas.bind('<B1-Motion>', draw_line)
    tCanvas.bind('<ButtonRelease-1>', reset_points)
    canvas = tCanvas
    # Create the label for displaying the predictions
    prediction_label = tk.Label(tRoot, text="", font=("Arial", 16))
    prediction_label.pack()
    # Create the clear and submit buttons
    clear_button = tk.Button(tRoot, text="Clear", command=lambda: clear_canvas(tCanvas))
    clear_button.pack(side="left", fill="x")
    submit_button = tk.Button(tRoot, text="Submit", command=lambda: submit(tRoot,tCanvas, prediction_label, o_models))
    submit_button.pack(side="left", fill="x")

    createMenus(tRoot)
    # Start the main loop
    tRoot.mainloop()

def createMenus(tRoot):
    global ui_model_names
    # Create the dropdown menus for selecting the models
    o_model_names = ["Not Simple Model", "MLP", "Simple CNN", "CNN", "Empty"]
    # Create a list of StringVar objects to store the selected model names
    model_names = []
    for i in range(len(o_model_names) - 1):
        model_name = tk.StringVar(tRoot)
        model_name.set(o_model_names[i])
        model_names.append(model_name)
    ui_model_names = model_names

    # Create the dropdown menus
    model_menus = []
    for model_name in model_names:
        model_menu = tk.OptionMenu(tRoot, model_name, *o_model_names)
        model_menu.pack(side="left", fill="x")
        model_menus.append(model_menu)

# Create the main window
tRoot = tk.Tk()
predictionWindow(tRoot)


def plot_history(history_list, tRoot = None, val_loss_list = None, val_acc_list = None):
    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2)
    selected_model_names = [ui_model_names[i] for i in selected_model_indices]

    # Plot the training loss for all models in the first subplot
    for i, history in enumerate(history_list):
        axs[0, 0].plot(history.history['loss'], label=selected_model_names[i].get())
    axs[0, 0].set_title('Model loss')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].legend()

    # Plot the validation loss for all models in the second subplot
    for i, history in enumerate(history_list):
        axs[0, 1].plot(history.history['val_loss'], label=selected_model_names[i].get())
    axs[0, 1].set_title('Model loss on validation data')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_xlabel('Model')
    axs[0, 1].legend()

    # Plot the validation accuracy for all models in the third subplot
    for i, history in enumerate(history_list):
        axs[1, 0].plot(history.history['val_accuracy'], label=selected_model_names[i].get())
    axs[1, 0].set_title('Model accuracy on validation data')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].set_xlabel('Model')
    axs[1, 0].legend()

    # Plot the predictions for all models in the fourth subplot
    for i, history in enumerate(history_list):
        axs[1, 1].plot(history.history['accuracy'], label=selected_model_names[i].get())
    axs[1, 1].set_title('Model Accuracy on training data')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].set_xlabel('Model')
    axs[1, 1].legend()

    if tRoot != None:
        canvas = FigureCanvasTkAgg(fig, master=tRoot)
        canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')
        tRoot.grid_columnconfigure(0, weight=1)
        tRoot.grid_rowconfigure(1, weight=1)
        tRoot.mainloop()
    else:
        plt.show()

def plot_predictions(predictions_list, tRoot=None):
    # Plot the predictions for all models
    selected_model_names = [ui_model_names[i] for i in selected_model_indices]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, predictions in enumerate(predictions_list):
        ax.scatter(range(len(predictions)), predictions[:, 0], label=selected_model_names[i].get())
    ax.title('Model predictions')
    ax.ylabel('Prediction')
    ax.xlabel('Sample index')
    ax.legend()
    if tRoot != None:
        canvas = FigureCanvasTkAgg(fig, master=tRoot)
        canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')
        tRoot.grid_columnconfigure(0, weight=1)
        tRoot.grid_rowconfigure(1, weight=1)
        tRoot.mainloop()
    else:
        plt.show()
# print(history_list)
# Plot the results
tRoot = tk.Tk()
plot_history(history_list, tRoot)
print(predictions_list)
tRoot = tk.Tk()
plot_predictions(predictions_list, tRoot)