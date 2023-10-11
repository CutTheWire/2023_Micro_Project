import platform
import subprocess

class TW:
    def __init__(self) -> None:
        self.result = "212NZMM047635"

    def __eq__(self, other) -> bool:
        if isinstance(other, TW):
            return self.result == other.result
        elif isinstance(other, str):
            return self.result == other
        else:
            return False
        
    def __call__(self):
        if platform.system() == 'Windows':
            try:
                result = subprocess.check_output('wmic bios get serialnumber').decode("utf-8")
                result = result.split("\r\r")[1].split("\n")[1].replace(" ", "").strip()
                return self == result
            except Exception as e:
                return e
        else:
            return 2