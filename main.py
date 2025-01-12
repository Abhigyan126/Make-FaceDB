import os
import uuid
import json
import pickle
import threading
import queue
from tkinter import Tk, Button, Label, filedialog, Text, ttk, Frame, messagebox
import face_recognition


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")
        self.root.geometry("800x600")
        
        # Initialize parameters
        self.known_faces = []
        self.face_uuids = []
        self.output_data = []
        self.processing = False
        self.current_folder = None
        self.supported_file_types = (".png", ".jpg", ".jpeg")

        # Queue for thread communication
        self.queue = queue.Queue()
        
        # Create main container
        self.create_widgets()
        
        # Load existing data if available
        self.load_parameters()
        
        # Start queue processing
        self.process_queue()

    def create_widgets(self):
        """Create and initialize all GUI widgets using grid"""
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(3, weight=1)

        # Title
        Label(self.root, text="Face Recognition Application", font=("Arial", 16)).grid(row=0, column=0, pady=10, columnspan=2)

        # Folder Frame
        folder_frame = Frame(self.root)
        folder_frame.grid(row=1, column=0, sticky='ew', padx=10)
        folder_frame.grid_columnconfigure(1, weight=1)

        Label(folder_frame, text="Selected Folder:", font=("Arial", 10)).grid(row=0, column=0, padx=5)
        self.folder_label = Label(folder_frame, text="No folder selected", font=("Arial", 10))
        self.folder_label.grid(row=0, column=1, sticky='w')

        # Button Frame
        button_frame = Frame(self.root)
        button_frame.grid(row=2, column=0, pady=10)

        self.select_button = Button(button_frame, text="Select Folder", command=self.select_folder, font=("Arial", 12))
        self.select_button.grid(row=0, column=0, padx=5)

        self.save_button = Button(button_frame, text="Save Parameters", command=self.save_parameters, font=("Arial", 12))
        self.save_button.grid(row=0, column=1, padx=5)

        # Progress Frame
        progress_frame = Frame(self.root)
        progress_frame.grid(row=3, column=0, sticky='ew', padx=10)
        progress_frame.grid_columnconfigure(0, weight=1)

        self.progress_label = Label(progress_frame, text="Progress: 0%", font=("Arial", 12))
        self.progress_label.grid(row=0, column=0, pady=5)

        self.progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate")
        self.progress_bar.grid(row=1, column=0, sticky='ew', pady=5)

        # Log Area Frame
        log_frame = Frame(self.root)
        log_frame.grid(row=4, column=0, sticky='nsew', padx=10, pady=5)
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)

        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.grid(row=0, column=1, sticky='ns')

        self.log_area = Text(log_frame, height=10, wrap="word", yscrollcommand=scrollbar.set, state="disabled")
        self.log_area.grid(row=0, column=0, sticky='nsew')
        scrollbar.config(command=self.log_area.yview)

    def process_queue(self):
        """Process messages from the queue"""
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg['type'] == 'log':
                    self.log_message(msg['message'])
                elif msg['type'] == 'progress':
                    self.update_progress(msg['current'], msg['total'])
                elif msg['type'] == 'complete':
                    self.processing = False
                    self.select_button.config(state="normal")
                    self.save_button.config(state="normal")
                    self.log_message("Processing complete!")
                    self._save_results(self.current_folder)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)

    def update_progress(self, current, total):
        """Update progress bar and label"""
        percentage = int((current / total) * 100)
        self.progress_bar['value'] = percentage
        self.progress_label.config(text=f"Progress: {percentage}%")

    def log_message(self, message):
        """Log messages to the text area"""
        self.log_area.config(state="normal")
        self.log_area.insert("end", message + "\n")
        self.log_area.config(state="disabled")
        self.log_area.see("end")

    def load_parameters(self):
        """Load saved parameters"""
        if os.path.exists("known_faces.pkl"):
            with open("known_faces.pkl", "rb") as file:
                self.known_faces, self.face_uuids = pickle.load(file)
            self.log_message("Loaded existing known faces and UUIDs.")
        else:
            self.log_message("No existing parameters found. Starting fresh.")

    def save_parameters(self):
        """Save parameters"""
        if self.processing:
            self.log_message("Cannot save while processing images.")
            return

        with open("known_faces.pkl", "wb") as file:
            pickle.dump((self.known_faces, self.face_uuids), file)
        self.log_message("Known faces and UUIDs saved successfully.")
        messagebox.showinfo("Save Parameters", "Parameters saved successfully!")

    def select_folder(self):
        """Select a folder"""
        if self.processing:
            self.log_message("Please wait for current processing to complete.")
            return

        folder_path = filedialog.askdirectory()
        if not folder_path:
            self.log_message("No folder selected.")
            return

        self.current_folder = folder_path
        self.folder_label.config(text=os.path.basename(folder_path))
        self.log_message(f"Selected folder: {folder_path}")

        self.processing = True
        self.output_data = []  # Reset output data
        self.select_button.config(state="disabled")
        self.save_button.config(state="disabled")
        threading.Thread(target=self.process_images, args=(folder_path,), daemon=True).start()

    def process_images(self, folder_path):
        """Process images in a folder"""
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(self.supported_file_types)]
        total_images = len(image_files)

        if total_images == 0:
            self.queue.put({'type': 'log', 'message': "No valid image files found."})
            self.queue.put({'type': 'complete', 'message': ''})
            return

        for i, image_name in enumerate(image_files, start=1):
            if not self.processing:
                break
            try:
                self._process_single_image(folder_path, image_name)
            except Exception as e:
                self.queue.put({'type': 'log', 'message': f"Error processing {image_name}: {e}"})
            self.queue.put({'type': 'progress', 'current': i, 'total': total_images})

        self.queue.put({'type': 'complete', 'message': ''})

    def _process_single_image(self, folder_path, image_name):
        """Process a single image"""
        image_path = os.path.join(folder_path, image_name)
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)

        if not face_locations:
            self.queue.put({'type': 'log', 'message': f"No faces detected in {image_name}. Skipping."})
            return

        face_encodings = face_recognition.face_encodings(image, face_locations)
        del image  # Free memory

        image_faces = []
        for encoding in face_encodings:
            face_id = self._process_face_encoding(encoding)
            image_faces.append(face_id)

        self.output_data.append({"image_name": image_name, "faces": image_faces})
        self.queue.put({'type': 'log', 'message': f"Processed {image_name}: {len(image_faces)} face(s) detected."})

    def _process_face_encoding(self, encoding):
        """Process a face encoding"""
        if self.known_faces:
            matches = face_recognition.compare_faces(self.known_faces, encoding)
            if True in matches:
                return self.face_uuids[matches.index(True)]

        face_id = str(uuid.uuid4())
        self.known_faces.append(encoding)
        self.face_uuids.append(face_id)
        return face_id

    def _save_results(self, folder_path):
        """Save results to a JSON file"""
        output_file = os.path.join(folder_path, "image_faces_data.json")
        with open(output_file, "w") as json_file:
            json.dump(self.output_data, json_file, indent=4)
        self.log_message(f"Face data saved to {output_file}")
        messagebox.showinfo("Processing Complete", f"Results saved to {output_file}")


def main():
    root = Tk()
    FaceRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
