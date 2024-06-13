import random
import zipfile
import tkinter as tk
from PIL import Image, ImageTk
import sounddevice as sd
import wave
from google.cloud import speech_v1p1beta1 as speech
import os
import subprocess

current_directory = os.path.dirname(os.path.abspath(__file__))
credentials_path = os.path.join(current_directory, 'key.json')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

accepted_commands = [
    "comandos",
    "clasificar vinos",
    "predecir precio del aguacate",
    "predecir masa corporal",
    "predecir precio de auto",
    "predecir precio del bitcoin",
    "recomendar una película",
]

file_path = os.path.join(current_directory, 'models', 'model2.pkl')
#reg_AveragePrice = joblib.load(file_path)

def button_clicked():
    messages.insert(tk.END, "Manten presionado para grabar!\n\n")
    print("Button clicked!\n")


def button_held():
    messages.insert(tk.END, "Grabando!\n\n")
    print("Recording!\n")
    global recording, audio_frames
    recording = True
    audio_frames = []


def on_button_press(event):
    global hold_job_id
    hold_job_id = root.after(500, button_held)


def on_button_release(event):
    root.after_cancel(hold_job_id)
    global recording
    recording = False
    if audio_frames:
        save_audio()
        process_command()


def save_audio():
    file_name = "command.wav"
    with wave.open(file_name, 'wb') as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)
        wave_file.setframerate(44100)
        wave_file.writeframes(b''.join(audio_frames))


def audio_callback(indata, frames, time, status):
    if recording:
        audio_frames.append(indata.copy())


def transcribe_audio():
    client = speech.SpeechClient().from_service_account_file('key.json')
    config = speech.RecognitionConfig(sample_rate_hertz=44100, enable_automatic_punctuation=True, language_code="es-ES")

    file_name = "command.wav"
    if not os.path.exists(file_name):
        return -1, None

    with open(file_name, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        command = result.alternatives[0].transcript
        message = "Commando: {}\n".format(command)
        messages.insert(tk.END, message)
        print(f"Mensaje: {message}")
        if command.lower() in accepted_commands:
            return 0, command

    return None, None


def process_command():
    response, command = transcribe_audio()

    if response == -1:
        messages.insert(tk.END, "No se encontró archivo de audio.\n\n")
        print("No audio file found.\n")
        return
    elif response is None:
        messages.insert(tk.END, "Comando desconocido!\n")
        print("Unknown command!\n")
        return

    messages.insert(tk.END, "Procesando comando!\n")
    print("Processing command!\n")

    command_functions = {
        "comandos": mostrar_comandos,
        "predecir precio del bitcoin": predecir_precio_bitcoin,
        "recomendar una película": recomendar_pelicula,
        "predecir precio de auto": predecir_precio_auto,
        "predecir precio del aguacate": predecir_precio_aguacate,
        "clasificar vinos": clasificar_vino,
        "predecir masa corporal": predecir_masa_corporal,

        # Add more commands and their corresponding functions here
    }

    command_function = command_functions.get(command)
    if command_function:
        command_function()
    else:
        messages.insert(tk.END, "Algo salió mal!\n")
        print("Unknown error!\n")
def predecir_precio_bitcoin():
    try:
        # Load the dataset
        file_path = r"C:\Users\Usuario\.kaggle\bitcoin.zip"

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            files = zip_ref.namelist()

            if 'bitcoin_price_Training - Training.csv' not in files:
                raise FileNotFoundError("No se encontró el archivo ZIP")

            with zip_ref.open('bitcoin_price_Training - Training.csv') as file:
                fat = [line.decode('utf-8').strip().split(',')[2] for line in file.readlines() if line]

        pred = random.choice(fat)

        # Display the prediction
        message = f"Predicción: $ {pred}\n"
        messages.insert(tk.END, f"{message}\n")
        print(message)

    except Exception as e:
        messages.insert(tk.END, f"Algo salió mal: {e}\n")
        print(f"Error predicting body fat price: {e}\n")

def predecir_masa_corporal():
    try:
        # Load the dataset
        file_path = r"C:\Users\Usuario\.kaggle\body_fat.zip"

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            files = zip_ref.namelist()

            if 'bodyfat.csv' not in files:
                raise FileNotFoundError("No se encontró el archivo ZIP")

            with zip_ref.open('bodyfat.csv') as file:
                fat = [line.decode('utf-8').strip().split(',')[1] for line in file.readlines() if line]

        pred = random.choice(fat)

        # Display the prediction
        message = f"Predicción: {pred}%\n"
        messages.insert(tk.END, f"{message}\n")
        print(message)

    except Exception as e:
        messages.insert(tk.END, f"Algo salió mal: {e}\n")
        print(f"Error predicting body fat price: {e}\n")

def predecir_precio_auto():
    try:
        # Load the dataset
        file_path = r"C:\Users\Usuario\.kaggle\cars.zip"

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            files = zip_ref.namelist()

            if 'car data.csv' not in files:
                raise FileNotFoundError("No se encontró el archivo ZIP")

            with zip_ref.open('car data.csv') as file:
                car = [line.decode('utf-8').strip().split(',')[2] for line in file.readlines() if line]

        pred = random.choice(car)

        # Display the prediction
        message = f"Predicción: $ {pred} k\n"
        messages.insert(tk.END, f"{message}\n")
        print(message)

    except Exception as e:
        messages.insert(tk.END, f"Algo salió mal: {e}\n")
        print(f"Error predicting body fat price: {e}\n")

def predecir_precio_aguacate():
    try:
        # Load the dataset
        file_path = r"C:\Users\Usuario\.kaggle\avocado_price.zip"

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            files = zip_ref.namelist()

            if 'avocado.csv' not in files:
                raise FileNotFoundError("No se encontró el archivo ZIP")

            with zip_ref.open('avocado.csv') as file:
                # Read each line, split by commas, and extract the price
                prices = [line.decode('utf-8').strip().split(',')[2] for line in file.readlines() if line]

        price = random.choice(prices)

        # Display the prediction
        message = f"Predicción: $ {price}\n"
        messages.insert(tk.END, f"{message}\n")
        print(message)

    except Exception as e:
        messages.insert(tk.END, f"Algo salió mal: {e}\n")
        print(f"Error predicting avocado price: {e}\n")

def recomendar_pelicula():
    try:
        # Load the dataset
        file_path = r"C:\Users\Usuario\.kaggle\IMDB-Dataset-movies.zip"

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            files = zip_ref.namelist()

            if 'movies.csv' not in files:
                raise FileNotFoundError("No se encontró el archivo ZIP")

            with zip_ref.open('movies.csv') as file:
                # Read each line, split by commas, and extract the movie name
                movie_names = [line.decode('utf-8').strip().split(',')[1] for line in file.readlines()]

        # Select a random movie name
        random_movie_name = random.choice(movie_names)

        # Display the recommendation
        message = f"Recomendación: {random_movie_name}\n"
        messages.insert(tk.END, f"{message}\n")
        print(message)

    except Exception as e:
        messages.insert(tk.END, f"Algo salió mal: {e}\n")
        print(f"Error recommending a movie: {e}\n")

def clasificar_vino():
    try:
        # Load the dataset
        file_path = r"C:\Users\Usuario\.kaggle\wine_quality.zip"

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            files = zip_ref.namelist()

            if 'winequalityN.csv' not in files:
                raise FileNotFoundError("No se encontró el archivo ZIP")

            with zip_ref.open('winequalityN.csv') as file:
                # Read each line, split by commas, and extract the price
                wine = [line.decode('utf-8').strip().split(',')[0] for line in file.readlines() if line]

        type = random.choice(wine)

        # Display the prediction
        message = f"Tipo: {type}\n"
        messages.insert(tk.END, f"{message}\n")
        print(message)

    except Exception as e:
        messages.insert(tk.END, f"Algo salió mal: {e}\n")
        print(f"Error predicting avocado price: {e}\n")

def mostrar_comandos():
    messages.insert(tk.END, "\nLista de comandos\n\n")
    print("Showing all commands\n")
    for command in accepted_commands:
        messages.insert(tk.END, f"{command}\n")

def execute_recognition_face():
    subprocess.Popen(['python', 'recognition_face.py'])

def open_link(url):
    import webbrowser
    webbrowser.open(url)

def main():
    global root, messages, recording, audio_frames
    recording = False
    audio_frames = []

    # Create the main window
    root = tk.Tk()
    root.title("AI Assistant")

    # Create a text area for messages
    messages = tk.Text(root, height=20, width=50)
    messages.pack(side=tk.LEFT, fill=tk.Y)

    # Load the image
    image_path = "src/bender.png"
    image = Image.open(image_path)

    # Convert the image to a tkinter-compatible photo image
    tk_image = ImageTk.PhotoImage(image)

    # Create a label with the image
    label = tk.Label(root, image=tk_image)
    label.pack(side=tk.LEFT)

    # Load and resize the button image
    button_image_path = "src/push_button.png"
    button_image = Image.open(button_image_path)
    button_image = button_image.resize((50, 50))  # Resize the image to 50x50
    tk_button_image = ImageTk.PhotoImage(button_image)

    # Create a button with the resized image
    button = tk.Button(root, image=tk_button_image, borderwidth=0, highlightthickness=0)
    button.image = tk_button_image  # Keep a reference to the image to prevent garbage collection

    # Position the button
    window_width = 800
    window_height = 600
    button_width = 50
    button_height = 50
    button_x = (window_width - button_width) // 2 + 410
    button_y = (window_height - button_height) // 2
    button.place(x=button_x, y=button_y)

    # Bind events to the button
    button.bind("<ButtonPress>", on_button_press)
    button.bind("<ButtonRelease>", on_button_release)

    # Create a button to execute recognition_face.py
    execute_button = tk.Button(
        root,
        text="FaceID",
        command=execute_recognition_face,
        font=("Helvetica", 14),
        bg="#4CAF50",
        fg="white",
        activebackground="#45a049",
        relief="raised",
        borderwidth=2,
        padx=10,
        pady=5
    )
    execute_button.pack(side=tk.BOTTOM, pady=20)

    # Create a button to open the sound link
    sound_button = tk.Button(
        root,
        text="Sonido",
        command=lambda: open_link(
            "https://colab.research.google.com/drive/1tgQpp24LAFfLEKeoPdR31DObuyV9yH4L?usp=sharing"),
        font=("Helvetica", 14),
        bg="#4CAF50",
        fg="white",
        activebackground="#45a049",
        relief="raised",
        borderwidth=2,
        padx=10,
        pady=5
    )
    sound_button.pack(side=tk.BOTTOM, pady=10)

    # Create a button to open the weapons link
    weapons_button = tk.Button(
        root,
        text="Armas",
        command=lambda: open_link(
            "https://colab.research.google.com/drive/1XiY8RErxH3npw1lmTdV5VSvLqu9Dwieq?usp=sharing"),
        font=("Helvetica", 14),
        bg="#4CAF50",
        fg="white",
        activebackground="#45a049",
        relief="raised",
        borderwidth=2,
        padx=10,
        pady=5
    )
    weapons_button.pack(side=tk.BOTTOM, pady=10)

    # Give directions to the users
    messages.insert(tk.END, "Mantén presionado el botón y di \"comandos\" para ver la lista de posibilidades\n\n")

    # Start audio stream
    with sd.InputStream(callback=audio_callback, channels=1, dtype='int16', samplerate=44100):
        # Start the tkinter main loop
        root.mainloop()

if __name__ == '__main__':
    main()

