import json
import numpy as np

class SettingReader:
    def __init__(self,filename="setting.json"):
        with open(filename, 'r', encoding='UTF-8') as settings_file:
            settings_data = json.load(settings_file)
            self.fp = np.float32 if settings_data["fp"] == "np.float32" else float


