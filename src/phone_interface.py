import requests
import time

# 1. REPLACE with the URL from your phyphox app
# Make sure to include "http://" and the port number
base_url = "http://10.0.0.3:8080"

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

print(f"Polling phyphox at: {query_url}")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        try:
            # 3. Make the HTTP GET request
            response = requests.get(query_url)
            response.raise_for_status()  # Raise an exception for bad status codes

            # 4. Parse the JSON response
            data = response.json()

            # 5. Extract the data
            # The data is in: data['buffer'][SENSOR_NAME]['buffer'][0]
            ax = data['buffer']['accX']['buffer'][0]
            ay = data['buffer']['accY']['buffer'][0]
            az = data['buffer']['accZ']['buffer'][0]

            # gx = data['buffer']['gyroX']['buffer'][0]
            # gy = data['buffer']['gyroY']['buffer'][0]
            # gz = data['buffer']['gyroZ']['buffer'][0]

            # 6. Print the data to the console
            # \r and end='' make it update on a single line
            # print(
            #     f"\rAccel: (X:{ax: >6.2f}, Y:{ay: >6.2f}, Z:{az: >6.2f}) | Gyro: (X:{gx: >6.2f}, Y:{gy: >6.2f}, Z:{gz: >6.2f})",
            #     end="")
            print(f"\rAccel: (X:{ax: >6.2f}, Y:{ay: >6.2f}, Z:{az: >6.2f})", end="")

        except requests.exceptions.ConnectionError:
            print(f"\nError: Could not connect to {base_url}.")
            print("Is phyphox running and 'Allow remote access' enabled?")
            print("Are your phone and computer on the same Wi-Fi?")
            break  # Exit the loop on connection error

        except (KeyError, IndexError) as e:
            # This happens if a sensor name is wrong or the buffer is empty
            print(f"\nError: Data not found. {e}")
            print("Make sure the experiment you are running in phyphox")
            print(f"provides all these buffers: {sensors}")
            print("Waiting...")
            time.sleep(2.0)  # Wait longer before retrying

        # Poll rate (e.g., 10 times per second)
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping data collection.")