import pyautogui
import time

# --- CONFIGURATION ---
# Map your Model's Output Labels (keys) to PyAutoGUI Key Names (values)
# PyAutoGUI keys: 'left', 'right', 'up', 'down', 'space', 'enter', 'a', 'b', etc.
KEY_MAPPING = {
    "Walk": "space",
    "Walk-circle": "space",
    "Run": "space",
    "Walk-backwards": "space",
    "Push-up": "space",
    "Jump": "space",
    "Pick": "space",
    "Stair-up": "space",
    "Stair-down": "space",
    "Sit-up": "space",
    "Stand": None             # Do nothing
}

# 1. Set a cooldown duration (in seconds)
# Adjust this: 0.5 means the key can only be pressed once every half second.
COOLDOWN_DURATION = 0.3

# 2. Create a dictionary to track the last time an action occurred
last_execution_times = {}

def execute_action(predicted_label):
    """
    Takes the label from the model and presses the corresponding key.
    """
    # 1. Look up the key
    key_to_press = KEY_MAPPING.get(predicted_label)

    # 2. Check if valid
    if key_to_press is None:
        print(f"Action '{predicted_label}' has no key bind.")
        return

    current_time = time.time()

    # Get the last time THIS specific label was executed (default to 0 if never)
    last_time = last_execution_times.get("Jump", 0)

    # 3. Press the key
    if (current_time - last_time) > COOLDOWN_DURATION:
        print(f"Binder: Pressing '{key_to_press}' for action '{predicted_label}'")

        pyautogui.press(key_to_press)

        # 4. Update the timestamp for this action
        last_execution_times["Jump"] = current_time
