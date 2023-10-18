import wmi

cpu_info = ""
c = wmi.WMI()
processors = c.Win32_Processor()
for processor in processors:
    cpu_info = processor.ProcessorId
    break

print(cpu_info)

drive = "C"
c = wmi.WMI()
logical_disks = c.Win32_LogicalDisk(DeviceID=drive + ":")
for disk in logical_disks:
    volume_serial = disk.VolumeSerialNumber
    break

print(volume_serial)

unique_id = cpu_info + volume_serial
print(unique_id)
