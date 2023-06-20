import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

# Create a Tkinter window
window = tk.Tk()
window.title("Object and Lane Detection")

# Create a canvas to display the processed frames
canvas = tk.Canvas(window, width=800, height=600)
canvas.pack()

# OpenCV video capture
cap = cv2.VideoCapture(0)  # Change the parameter to the desired video source

def process_frame():
    _, frame = cap.read()
    
    # Process the frame using your code
    processed_frame = process_image(frame)
    
    # Convert the processed frame to PIL Image format
    img = Image.fromarray(processed_frame)
    
    # Resize the image to fit the canvas
    img = img.resize((800, 600))
    
    # Convert the PIL Image to Tkinter-compatible ImageTk format
    img_tk = ImageTk.PhotoImage(img)
    
    # Update the canvas with the new image
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    
    # Schedule the next frame processing
    window.after(1, process_frame)

# Start processing the frames
process_frame()

# Start the Tkinter event loop
window.mainloop()

# Release the video capture
cap.release()
