import requests
import time
import threading
import wx

class Collect_Data_Thread(threading.Thread):
    def __init__(self, url, sensors, callback):
        super().__init__()
        self.query_url = url
        self.sensors = sensors
        self.callback = callback
        self.stop_event = threading.Event()
        self.daemon = True

    def run(self):
        print(f"Polling phyphox at: {self.query_url}")

        while not self.stop_event.is_set():
            try:
                # 3. Make the HTTP GET request
                response = requests.get(self.query_url)
                response.raise_for_status()  # Raise an exception for bad status codes

                # 4. Parse the JSON response
                data = response.json()

                # 5. Extract the data
                # The data is in: data['buffer'][SENSOR_NAME]['buffer'][0]
                num_data = {}
                for s in self.sensors:
                    num_data[s] = data['buffer'][s]['buffer'][0]

                wx.CallAfter(self.callback, num_data)

                # 6. Print the data to the console
                # \r and end='' make it update on a single line
                print(f"\rAccel: (X:{num_data[self.sensors[0]]: >6.2f}, "
                      f"Y:{num_data[self.sensors[1]]: >6.2f}, "
                      f"Z:{num_data[self.sensors[2]]: >6.2f})", end="")

            except requests.exceptions.ConnectionError:
                wx.CallAfter(self.callback,
                             {"Error: Could not connect to " + self.query_url +
                              "\nPhyphox running? 'Allow remote access' enabled? Same Wi-Fi?"})
                self.stop_event.set()

            except (KeyError, IndexError) as e:
                # This happens if a sensor name is wrong or the buffer is empty
                print(f"\nError: Data not found. {e}")
                print(f"Make sure phyphox provides all these buffers: {self.sensors}")
                wx.CallAfter(self.callback, {"Error: wrong data format"})
                print("Waiting...")
                time.sleep(2.0)  # Wait longer before retrying

            # Poll rate (e.g., 10 times per second)
            self.stop_event.wait(0.1)

        print("\nStopping data collection.")

    def stop(self):
        self.stop_event.set()