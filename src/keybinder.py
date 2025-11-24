import pyautogui

# --- CONFIGURATION ---
# Map your Model's Output Labels (keys) to PyAutoGUI Key Names (values)
# PyAutoGUI keys: 'left', 'right', 'up', 'down', 'space', 'enter', 'a', 'b', etc.
KEY_MAPPING = {
    "Walk": "space",
    "Walk-circle": "space",
    "Run": "space",
    "Walk-backward": "space",
    "Push-up": "space",
    "Jump": "space",       # Simulates Spacebar
    "Stand": None             # Do nothing
}

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

    # 3. Press the key
    print(f"Binder: Pressing '{key_to_press}' for action '{predicted_label}'")
    pyautogui.press(key_to_press)