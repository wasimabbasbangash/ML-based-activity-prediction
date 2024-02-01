import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import subprocess

def run_script(script_name, file_path):
    result = subprocess.run(["python", script_name, file_path], capture_output=True, text=True)
    output.insert(tk.END, f"=== Output of {script_name} ===\n")
    output.insert(tk.END, result.stdout)
    output.insert(tk.END, "\n\n")

def upload_file():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("XES files", "*.xes")])
    if file_path:
        classifier_button_frame.pack(side=tk.BOTTOM, pady=2)  # Pack the frame containing the classifier buttons
    else:
        messagebox.showwarning("File not loaded", "No file was selected.")

def run_random_classifier():
    run_script("random_classifier.py", file_path)

def run_ngram_classifier():
    run_script("ngram_classifier.py", file_path)

# Create the main window
root = tk.Tk()
root.title("Presentation 2 by Waseem")
root.state('zoomed')  # Make the window full-screen

# Load and display the logo image
image = Image.open("logo.jpg")
# Resize the logo to fit the window size
image = image.resize((int(root.winfo_screenwidth() * 0.5), int(root.winfo_screenheight() * 0.2)), Image.Resampling.LANCZOS) 
logo = ImageTk.PhotoImage(image)
logo_label = tk.Label(root, image=logo)
logo_label.pack(pady=20)

# Create a label with instructions
label = tk.Label(root, text="Please upload your dataset in Xes format.", padx=10, pady=10)
label.pack(side=tk.TOP, pady=16)

# Create a file upload button and place it at the top
upload_button = tk.Button(root, text="Upload File", command=upload_file, padx=10, pady=5)
upload_button.pack(side=tk.TOP, pady=2)

# Create a frame for classifier buttons
classifier_button_frame = tk.Frame(root)
random_classifier_button = tk.Button(classifier_button_frame, text="Run Random Classifier", command=run_random_classifier, padx=10, pady=5)
random_classifier_button.pack(side=tk.LEFT, padx=10)
ngram_classifier_button = tk.Button(classifier_button_frame, text="Run Ngram Classifier", command=run_ngram_classifier, padx=10, pady=5)
ngram_classifier_button.pack(side=tk.LEFT, padx=10)

# Create a text widget to display output
output = scrolledtext.ScrolledText(root, width=100, height=10)
output.pack(expand=True, fill=tk.BOTH)

# Run the application
root.mainloop()
