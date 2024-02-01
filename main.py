import tkinter as tk
from tkinter import filedialog, messagebox, Frame, Text, Scrollbar
from PIL import Image, ImageTk
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from PIL import Image, ImageTk
import threading

def display_image(image_path, output_frame):
    for widget in output_frame.winfo_children():
        widget.destroy()

    try:
        img = Image.open(image_path)
        img = img.resize((600, 450), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)

        img_label = tk.Label(output_frame, image=photo)
        img_label.image = photo
        img_label.pack()
    except FileNotFoundError:
        error_label = tk.Label(output_frame, text=f"File not found: {image_path}")
        error_label.pack()

def display_classification_report(report, output_frame):
    # Clear previous output in the text frame
    for widget in output_frame.winfo_children():
        widget.destroy()

    # Create a Text widget to display the classification report
    text_widget = Text(output_frame, wrap='word', font=('TkDefaultFont', 11))
    text_widget.pack(expand=True, fill='both', side='left')
    text_widget.insert('1.0', report)

    # Disable the widget so users can't edit the report
    text_widget.config(state='disabled')

def run_script_in_thread(script_name, file_path, image_frame, text_frame):
    def target():
        result = subprocess.run(["python", script_name, file_path], capture_output=True, text=True)

        # Assuming the script saves a plot as an image, display the plot
        plot_image_path = 'plot.png'
        
        # Schedule the display_image function to run on the main thread for image
        image_frame.after(0, display_image, plot_image_path, image_frame)
        
        # Schedule the display_classification_report function to run on the main thread for text
        text_frame.after(0, display_classification_report, result.stdout, text_frame)

    thread = threading.Thread(target=target)
    thread.start()

def upload_file():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("XES files", "*.xes")])
    if file_path:
        classifier_button_frame.pack(side=tk.BOTTOM, pady=2)
    else:
        messagebox.showwarning("File not loaded", "No file was selected.")

def run_random_classifier():
    run_script_in_thread("random_classifier.py", file_path, image_frame, text_frame)

def run_ngram_classifier():
    run_script_in_thread("ngram_classifier.py", file_path, image_frame, text_frame)


# Create the main window
root = tk.Tk()
root.title("Presentation 2 by Waseem")
root.state('zoomed')  # Make the window full-screen

# Load and display the logo image
image = Image.open("logo.jpg")
image = image.resize((int(root.winfo_screenwidth() * 0.5), int(root.winfo_screenheight() * 0.2)), Image.Resampling.LANCZOS)
logo = ImageTk.PhotoImage(image)
logo_label = tk.Label(root, image=logo)
logo_label.pack(pady=20)

# Create a label with instructions
label = tk.Label(root, text="Please upload your dataset in Xes format.", padx=10, pady=10)
label.pack(side=tk.TOP, pady=16)

# Create a file upload button
upload_button = tk.Button(root, text="Upload File", command=upload_file, padx=10, pady=5)
upload_button.pack(side=tk.TOP, pady=2)

# Create a frame for classifier buttons
classifier_button_frame = tk.Frame(root)
random_classifier_button = tk.Button(classifier_button_frame, text="Run Random Classifier", command=run_random_classifier, padx=10, pady=5)
random_classifier_button.pack(side=tk.LEFT, padx=10)
ngram_classifier_button = tk.Button(classifier_button_frame, text="Run Ngram Classifier", command=run_ngram_classifier, padx=10, pady=5)
ngram_classifier_button.pack(side=tk.LEFT, padx=10)

# Create a frame for output
output_frame = Frame(root)
output_frame.pack(fill='both', expand=True)

# Create subframes within the output frame
image_frame = Frame(output_frame, width=root.winfo_screenwidth() // 2)
image_frame.pack(side='left', fill='both', expand=True)
text_frame = Frame(output_frame, width=root.winfo_screenwidth() // 2)
text_frame.pack(side='right', fill='both', expand=True)

# Run the application
root.mainloop()
