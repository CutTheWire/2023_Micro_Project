import platform
import subprocess

def get_sm_info():
    try:
        # Get system information using the platform module
        system_info = platform.system()
        system_name = platform.node()
        system_version = platform.version()
        system_architecture = platform.architecture()
        system_serial_number = subprocess.check_output('wmic bios get serialnumber').decode("utf-8")
        system_serial_number_list = system_serial_number.split('\n')

        
        print("System Information:")
        print("System Name:", system_name)
        print("System Version:", system_version)
        print("System Architecture:", system_architecture)
        print("MB Number:", system_serial_number_list[1])

    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    print("Getting System Management (SM) Information:")
    get_sm_info()
