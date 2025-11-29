import requests
import time
import threading
import wx


class Collect_Data_Thread(threading.Thread):
    """
    Background thread that polls a phyphox /get endpoint and sends data
    back to the GUI using a callback.
    Full query URL: "http://yoururlfromphyphox/get?accX&accY&accZ" (should appear interminal)
    """

    def __init__(self, url, sensors, callback):
        super().__init__()
        self.query_url = url
        self.sensors = sensors
        self.callback = callback
        self.stop_event = threading.Event()
        self.daemon = True  # do not block program exit

    def run(self):
        print(f"Polling phyphox at: {self.query_url}")

        while not self.stop_event.is_set():
            try:
                # 1. Request latest sensor values from phyphox
                resp = requests.get(self.query_url, timeout=2.0)
                resp.raise_for_status()

                data_json = resp.json()

                # 2. Determine where the sensor objects live:
                container = data_json.get("buffer", None)
                if isinstance(container, dict):
                    sensor_root = container
                else:
                    sensor_root = data_json

                if not isinstance(sensor_root, dict):
                    raise ValueError(f"Unexpected JSON format: {data_json}")

                # 3. Extract latest values for each requested sensor
                values = {}

                for sensor in self.sensors:
                    if sensor not in sensor_root:
                        raise KeyError(
                            f"Sensor '{sensor}' not found in response. "
                            f"Available keys: {list(sensor_root.keys())}"
                        )

                    sensor_obj = sensor_root[sensor]
                    buffer_vals = sensor_obj.get("buffer", None)

                    # If buffer is nested differently, try a bit more defensive logic
                    if buffer_vals is None:
                        # Maybe the object itself is a list (rare)
                        buffer_vals = sensor_obj

                    if not isinstance(buffer_vals, (list, tuple)) or len(buffer_vals) == 0:
                        raise IndexError(f"No numeric data found for sensor '{sensor}'")

                    # Take the most recent value
                    last_val = buffer_vals[-1]
                    values[sensor] = float(last_val)

                # 4. Send clean dict to GUI on the main thread
                wx.CallAfter(self.callback, values)

            except requests.RequestException as e:
                # Networking problems: timeout, connection refused, etc.
                err_msg = (
                    f"Request to phyphox failed: {e}. "
                    "Check that phone and PC are on the same network, "
                    "remote access is enabled, and the URL is correct."
                )
                print("\n" + err_msg)
                wx.CallAfter(self.callback, {"error": err_msg})
                # Wait a bit longer before next attempt
                time.sleep(2.0)

            except (KeyError, IndexError, ValueError, TypeError) as e:
                # Problems with data format or missing buffers
                err_msg = (
                    f"Data format error: {e}. "
                    f"Make sure phyphox provides all these buffers: {self.sensors}"
                )
                print("\n" + err_msg)
                # IMPORTANT: send a dict, not a set
                wx.CallAfter(self.callback, {"error": "wrong data format"})
                print("Waiting before retrying...")
                time.sleep(2.0)
            # 5. Normal polling rate (unless we already slept due to error)
            if not self.stop_event.is_set():
                # Poll ~10 times per second
                self.stop_event.wait(0.1)

        print("\nStopping data collection.")

    def stop(self):
        """Signal the thread to stop on the next loop."""
        self.stop_event.set()
