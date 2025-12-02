import wx
from phone_interface import Collect_Data_Thread
from ml_models.random_forest import train_model, save_model, load_model, load_demo_model
import numpy as np
from keybinder import execute_action
from windowing import SlidingWindow
from constants import WINDOW_SIZE, WINDOW_STEP_SIZE, WINDOW_PADDING, WINDOW_MAX_BUFFER_SIZE

# Use the 3-class DEMO model (Stand / Walk / Jump) if True.
# Set False to test with the full 18-class model
USE_DEMO_MODEL = True
# 1. REPLACE with the URL from your phyphox app
# Make sure to include "http://" and the port number
base_url =  "http://192.168.0.36:8080"

# Thai phyphox url: "http://192.168.0.27:8080"
#default: "http://10.0.0.4:8080"

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

# Number of samples per window to use for one prediction
# this should match whatever you eventually use to train the model
SAMPLES_PER_WINDOW = WINDOW_SIZE

# Windowing configuration (imported from constants.py)
# Step size for sliding window (stride between predictions)
# If WINDOW_STEP_SIZE == SAMPLES_PER_WINDOW: non-overlapping windows
# If WINDOW_STEP_SIZE < SAMPLES_PER_WINDOW: overlapping windows (more frequent predictions)
SLIDING_WINDOW_STEP = WINDOW_STEP_SIZE
SLIDING_WINDOW_PADDING = WINDOW_PADDING
SLIDING_WINDOW_MAX_BUFFER = WINDOW_MAX_BUFFER_SIZE

app = wx.App()

class UIFrame(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=wx.Size(1080, 720))
        panel = wx.Panel(self)

        # LOAD THE MODEL 
        self.model = None
        try:
            if USE_DEMO_MODEL:
                self.model = load_demo_model()
                print("DEMO model (Stand / Walk / Jump) loaded successfully!")
            else:
                self.model = load_model()
                print("Full 18-class model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

        # Initialize sliding window for real-time predictions
        self.window = SlidingWindow(
            window_size=SAMPLES_PER_WINDOW,
            step_size=SLIDING_WINDOW_STEP,
            padding_strategy=SLIDING_WINDOW_PADDING,
            max_buffer_size=SLIDING_WINDOW_MAX_BUFFER,
        )
        print(f"Initialized sliding window: {self.window}")

        # UI ELEMENTS 
        title_label = wx.StaticText(panel, label="Motion Controller Demo")
        title_font = title_label.GetFont()
        title_font.PointSize += 6
        title_font.MakeBold()
        title_label.SetFont(title_font)

        self.data_label = wx.StaticText(panel, label="Click 'Start' to collect data.")
        font = self.data_label.GetFont()
        font.PointSize += 2
        self.data_label.SetFont(font)

        self.result_label = wx.StaticText(panel, label="Waiting for gesture...")
        result_font = self.result_label.GetFont()
        result_font.PointSize = 24
        result_font.MakeBold()
        self.result_label.SetFont(result_font)

        self.data_thread: Collect_Data_Thread | None = None

        self.start_button = wx.Button(
            panel, label="Start Controller", size=wx.Size(200, 100)
        )
        self.start_button.Bind(wx.EVT_BUTTON, self.on_start_click)

        self.stop_button = wx.Button(
            panel, label="Stop Controller", size=wx.Size(200, 100)
        )
        self.stop_button.Bind(wx.EVT_BUTTON, self.on_stop_click)
        self.stop_button.Disable()

        self.Bind(wx.EVT_CLOSE, self.on_close)

        # Layout
        outer_sizer = wx.BoxSizer(wx.VERTICAL)
        outer_sizer.Add(title_label, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 10)
        outer_sizer.Add(self.start_button, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)
        outer_sizer.Add(self.stop_button, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)
        outer_sizer.Add(self.data_label, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 20)
        outer_sizer.Add(self.result_label, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 10)

        panel.SetSizer(outer_sizer)
        self.Center()

    # Feature extraction: must match training
    def extract_features(self, buffer: list[list[float]]) -> np.ndarray:
        """
        Converts a buffer of raw [x, y, z] values into one feature row.

        Currently: for each axis (X, Y, Z) compute:
          mean, std, min, max, median, energy  ->  6 * 3 = 18 features

        IMPORTANT: Whatever you do here must be mirrored in the training
        script if you want the controller model to work perfectly.
        """
        data_array = np.array(buffer)  # shape (samples, 3)

        features = []

        for axis_idx in range(3):
            axis_data = data_array[:, axis_idx]

            _mean = np.mean(axis_data)
            _std = np.std(axis_data)
            _min = np.min(axis_data)
            _max = np.max(axis_data)
            _median = np.median(axis_data)
            _energy = np.mean(axis_data**2)  # mean squared energy

            features.extend([_mean, _std, _min, _max, _median, _energy])

        # shape: (1, 18)
        return np.array(features).reshape(1, -1)

    # Start / Stop
    def on_start_click(self, event):
        """Called when the 'Start Controller' button is clicked."""
        if self.data_thread and self.data_thread.is_alive():
            print("Thread already running.")
            return

        print("Starting data collection thread...")
        self.window.reset()  # Clear window buffer for fresh predictions

        self.data_thread = Collect_Data_Thread(
            url=query_url,
            sensors=sensors,
            callback=self.on_new_data,
        )
        self.data_thread.start()

        self.start_button.Disable()
        self.stop_button.Enable()
        self.data_label.SetLabel("Connecting to phyphox...")

    def on_stop_click(self, event):
        """Called when the 'Stop Controller' button is clicked."""
        if self.data_thread and self.data_thread.is_alive():
            print("Stopping data collection thread...")
            self.data_thread.stop()
        else:
            print("Thread not running.")

        self.start_button.Enable()
        self.stop_button.Disable()
        self.data_label.SetLabel("Data collection stopped.")

    # Data callback from phone thread
    def on_new_data(self, data: dict):
        """
        Callback from Collect_Data_Thread.
        Runs on the main UI thread.
        
        Each incoming sample is added to the sliding window.
        When a complete window is ready, a prediction is made automatically.
        """
        if "Error" in data:
            self.data_label.SetLabel(f"Error: {data['error']}. Check console.")
            self.start_button.Enable()
            self.stop_button.Disable()
            return

        try:
            x = data["accX"]
            y = data["accY"]
            z = data["accZ"]

            # Show raw values for debugging
            self.data_label.SetLabel(
                f"X: {x:.2f} | Y: {y:.2f} | Z: {z:.2f} | "
                f"Buffer: {self.window.buffer_size}/{SAMPLES_PER_WINDOW}"
            )

            # Add sample to sliding window
            self.window.add_sample([x, y, z])

            # If we have enough samples for a prediction window, make one
            if self.window.is_ready:
                self.make_prediction_from_window()
                self.window.advance()  # Slide window by step_size samples

        except KeyError:
            # If data keys are missing, silently ignore this packet
            pass

    # Prediction + key press
    def make_prediction_from_window(self):
        """
        Extract features from the current sliding window and make a prediction.
        
        This is called automatically when a window is ready in on_new_data.
        """
        if self.model is None:
            return

        try:
            # Get the current window as numpy array (window_size, 3)
            window_data = self.window.get_current_window()
            
            # Convert to list format for feature extraction
            window_list = window_data.tolist()
            
            # Extract features from the window
            processed_input = self.extract_features(window_list)
            
            # Get prediction
            prediction_index = self.model.predict(processed_input)[0]

            if USE_DEMO_MODEL:
                # DEMO model: we trained only Stand/Walk/Jump with indices 0..2
                class_names = {
                    0: "Stand",
                    1: "Walk",
                    2: "Jump",
                }
            else:
                # Full 18-class mapping
                class_names = {
                    0: "Stand",
                    1: "Sit",
                    2: "Talk-sit",
                    3: "Talk-stand",
                    4: "Stand-sit",
                    5: "Lay",
                    6: "Lay-stand",
                    7: "Pick",
                    8: "Jump",
                    9: "Push-up",
                    10: "Sit-up",
                    11: "Walk",
                    12: "Walk-backwards",
                    13: "Walk-circle",
                    14: "Run",
                    15: "Stair-up",
                    16: "Stair-down",
                    17: "Table-tennis",
                }

            result_text = class_names.get(prediction_index, "Unknown")

            # Update UI
            self.result_label.SetLabel(result_text)

            # Trigger mapped key if meaningful
            self.trigger_key_press(result_text)
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            self.result_label.SetLabel("Prediction Error")

    def trigger_key_press(self, action_name: str):
        """
        Map predicted action label to a key press (via keybinder).
        Usually skip "Unknown".
        """
        if action_name != "Unknown":
            execute_action(action_name)

    # Window close
    def on_close(self, event):
        """Ensure the thread is stopped when the window closes."""
        self.on_stop_click(None)
        self.Destroy()


mainFrame = UIFrame(None, "Motion Controller")
mainFrame.Show()
app.MainLoop()