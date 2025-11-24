import wx
from phone_interface import Collect_Data_Thread
from ml_models.random_forest import train_model, save_model, load_model
import numpy as np

# 1. REPLACE with the URL from your phyphox app
# Make sure to include "http://" and the port number
base_url = "http://10.0.0.4:8080"

# 2. Define the sensors you want to read.
# These are the "buffer" names in phyphox.
# For the "Accelerometer & Gyroscope" experiment, they are:
# accX, accY, accZ
# gyroX, gyroY, gyroZ
# sensors = ["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ"]
sensors = ["accX", "accY", "accZ"]

# Construct the full URL for the /get endpoint
# This asks phyphox for the latest value of each sensor
query_url = base_url + "/get?" + "&".join(sensors)

app = wx.App()

class UIFrame(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=wx.Size(1080, 720))
        panel = wx.Panel(self)

        # --- LOAD THE MODEL ---
        try:
            self.model = load_model()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

        # --- INITIALIZE BUFFER ---
        # This list will temporarily hold incoming data until we have enough to predict
        self.gesture_buffer = []

        label = wx.StaticText(panel, label="This is the beginning of the end.")

        # A label to display the real-time data
        self.data_label = wx.StaticText(panel, label="Click 'Start' to collect data.")
        font = self.data_label.GetFont()
        font.PointSize += 2
        self.data_label.SetFont(font)

        # Result Label (To show the prediction)
        self.result_label = wx.StaticText(panel, label="Waiting for gesture...")
        result_font = self.result_label.GetFont()
        result_font.PointSize = 24
        result_font.MakeBold()
        self.result_label.SetFont(result_font)

        self.data_thread = None
        self.start_button = wx.Button(panel, label="Start Controller", size=wx.Size(200, 100))
        self.start_button.Bind(wx.EVT_BUTTON, lambda event: self.on_start_click(event))
        self.stop_button = wx.Button(panel, label="Stop Controller", size=wx.Size(200, 100))
        self.stop_button.Bind(wx.EVT_BUTTON, self.on_stop_click)
        self.stop_button.Disable()  # Disable until started
        self.Bind(wx.EVT_CLOSE, self.on_close)  # Handle window close

        wrapper = wx.GridSizer(1)
        v_box = wx.BoxSizer(wx.VERTICAL)
        v_box.Add(self.start_button, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 10)
        v_box.Add(self.stop_button, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 10)
        v_box.Add(self.data_label, 1, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 20)
        v_box.Add(self.result_label, 1, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 20)
        wrapper.Add(v_box, 0, wx.ALIGN_CENTER)
        panel.SetSizer(wrapper)
        self.Center()

    def extract_features(self, buffer):
        """
        Converts a buffer of raw [x, y, z] values into a single row of features.
        Must match the training logic EXACTLY.
        """
        # Convert list of lists to a numpy array for easy math
        # Shape becomes (Samples, 3) -> e.g., (50, 3)
        data_array = np.array(buffer)

        features = []

        # Loop through the 3 columns (0=X, 1=Y, 2=Z)
        for axis_idx in range(3):
            axis_data = data_array[:, axis_idx]  # Get all rows for this axis

            # Calculate the stats
            _mean = np.mean(axis_data)
            _std = np.std(axis_data)
            _min = np.min(axis_data)
            _max = np.max(axis_data)
            _median = np.median(axis_data)

            # Energy is often defined as sum of squares, or mean of squares
            # Use whichever one you used in training!
            # Here is Mean Squared Energy:
            _energy = np.mean(axis_data ** 2)

            features.extend([_mean, _std, _min, _max, _median, _energy])

        # Reshape for the model: (1, 15) if you used 5 stats * 3 axes
        return np.array(features).reshape(1, -1)

    def on_start_click(self, event):
        """Called when the 'Start' button is clicked"""
        if self.data_thread and self.data_thread.is_alive():
            print("Thread already running.")
            return

        print("Starting data collection thread...")
        self.gesture_buffer = []

        # 6. Create and start the IMPORTED data collector thread
        # Pass the config and the callback function to it
        self.data_thread = Collect_Data_Thread(
            url=query_url, sensors=sensors, callback=self.on_new_data)
        self.data_thread.start()

        self.start_button.Disable()
        self.stop_button.Enable()
        self.data_label.SetLabel("Connecting to phyphox...")

    def on_stop_click(self, event):
        """Called when the 'Stop' button is clicked"""
        if not self.data_thread or not self.data_thread.is_alive():
            print("Thread not running.")
        else:
            print("Stopping data collection thread...")
            # 7. Tell the thread to stop
            self.data_thread.stop()

        self.start_button.Enable()
        self.stop_button.Disable()
        self.data_label.SetLabel("Data collection stopped.")

    def on_new_data(self, data):
        """
        This is the callback function!
        It runs SAFELY on the main UI thread.
        """
        if "Error" in data:
            self.data_label.SetLabel(f"Error: {data['error']}. Check console.")
            self.start_button.Enable()
            self.stop_button.Disable()
        else:
            # 1. Extract Raw Values
            try:
                x = data["accX"]
                y = data["accY"]
                z = data["accZ"]

                # Update debug label
                self.data_label.SetLabel(f"X: {x:.2f} | Y: {y:.2f} | Z: {z:.2f}")

                # 2. Add to Buffer
                self.gesture_buffer.append([x, y, z])

                # 3. Check if we have enough data to make a prediction
                samples = 3
                if len(self.gesture_buffer) >= samples:
                    self.make_prediction()
                    # Clear buffer to start listening for the next gesture
                    # (Or implement a sliding window if preferred)
                    self.gesture_buffer = []

            except KeyError:
                pass

    def make_prediction(self):
        """Prepares data and asks the model for an answer"""
        if not self.model:
            return

        processed_input = self.extract_features(self.gesture_buffer)
        # Make Prediction
        prediction_index = self.model.predict(processed_input)[0]

        # Map index to Name (Update this dictionary to match your classes)
        class_names = {0: "Stand", 1: "Sit", 2: "Talk-sit", 3: "Talk-stand",
                       4: "Stand-sit", 5: "Lay", 6: "Lay-stand", 7: "Pick",
                       8: "Jump", 9: "Push-up", 10: "Sit-up", 11: "Walk",
                       12: "Walk-backwards", 13: "Walk-circle", 14: "Run",
                       15: "Stair-up", 16: "Stair-down", 17: "Table-tennis"}
        result_text = class_names.get(prediction_index, "Unknown")

        # Update UI
        self.result_label.SetLabel(result_text)

    def on_close(self, event):
        """Ensure the thread is stopped when the window closes"""
        self.on_stop_click(None)  # Politely stop the thread
        self.Destroy()  # Close the window

mainFrame = UIFrame(None, "Motion Controller")
mainFrame.Show()
app.MainLoop()