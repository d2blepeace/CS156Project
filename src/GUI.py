import wx
from phone_interface import Collect_Data_Thread

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
        label = wx.StaticText(panel, label="This is the beginning of the end.")

        # A label to display the real-time data
        self.data_label = wx.StaticText(panel, label="Click 'Start' to collect data.")
        font = self.data_label.GetFont()
        font.PointSize += 2
        self.data_label.SetFont(font)

        self.data_thread = None
        self.start_button = wx.Button(panel, label="Start Controller", size=wx.Size(100, 100))
        self.start_button.Bind(wx.EVT_BUTTON, lambda event: self.on_start_click(event))
        self.stop_button = wx.Button(panel, label="Stop Controller", size=wx.Size(100, 100))
        self.stop_button.Bind(wx.EVT_BUTTON, self.on_stop_click)
        self.stop_button.Disable()  # Disable until started
        self.Bind(wx.EVT_CLOSE, self.on_close)  # Handle window close

        v_box = wx.BoxSizer(wx.VERTICAL)
        v_box.Add(self.start_button, 0, wx.ALL | wx.EXPAND, 10)
        v_box.Add(self.stop_button, 0, wx.ALL | wx.EXPAND, 10)
        v_box.Add(self.data_label, 1, wx.ALL | wx.CENTER | wx.EXPAND, 20)
        panel.SetSizer(v_box)
        self.Center()

    def on_start_click(self, event):
        """Called when the 'Start' button is clicked"""
        if self.data_thread and self.data_thread.is_alive():
            print("Thread already running.")
            return

        print("Starting data collection thread...")
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
            return

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
            # Dynamically build the label string from the sensors list
            label_parts = []
            for sensor in sensors:
                if sensor in data:
                    # Format as: "accX: -0.01"
                    label_parts.append(f"{sensor}: {data[sensor]: >6.2f}")

            # Join all parts with a separator
            self.data_label.SetLabel(" | ".join(label_parts))

    def on_close(self, event):
        """Ensure the thread is stopped when the window closes"""
        self.on_stop_click(None)  # Politely stop the thread
        self.Destroy()  # Close the window

mainFrame = UIFrame(None, "Motion Controller")
mainFrame.Show()
app.MainLoop()