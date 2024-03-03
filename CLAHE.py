
import cv2
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def enhance_images_with_clahe(input_image_path, output_folder, clip_limit=2.0, tile_grid_size=(8, 8), counter=None):
    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge([cl, a, b])
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    output_filename = f"hem{counter}.bmp"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, enhanced_img)

    return output_path

def process_and_display_images(input_folder, output_folder):
    counter = 1
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            input_image_path = os.path.join(input_folder, filename)
            enhanced_image_path = enhance_images_with_clahe(input_image_path, output_folder, counter=counter)
            display_processed_image(enhanced_image_path)
            counter += 1

def open_folder_dialog():
    input_folder_path = filedialog.askdirectory(title="Select Input Folder")
    if input_folder_path:
        output_folder_path = os.path.join(input_folder_path, "output")
        os.makedirs(output_folder_path, exist_ok=True)
        process_and_display_images(input_folder_path, output_folder_path)

def display_processed_image(image_path):
    processed_image = Image.open(image_path)
    processed_image.thumbnail((400, 400)) 

    tk_image = ImageTk.PhotoImage(processed_image)
    
    processed_image_label.config(image=tk_image)
    processed_image_label.image = tk_image

root = tk.Tk()
root.title("Image Enhancement with CLAHE")

select_folder_button = tk.Button(root, text="Select Input Folder", command=open_folder_dialog)
select_folder_button.pack(pady=10)

processed_image_label = tk.Label(root)
processed_image_label.pack(pady=10)

root.mainloop()
