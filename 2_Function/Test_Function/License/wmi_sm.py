import wmi
import tkinter as tk
import os

# CPU 정보 가져오기
cpu_info = ""
c = wmi.WMI()
processors = c.Win32_Processor()
for processor in processors:
    cpu_info = processor.ProcessorId
    break

# 드라이브 일련번호 가져오기
drive = "C"
c = wmi.WMI()
logical_disks = c.Win32_LogicalDisk(DeviceID=drive + ":")
for disk in logical_disks:
    volume_serial = disk.VolumeSerialNumber
    break

# unique_id 생성
unique_id = cpu_info + volume_serial

# Tkinter 창 생성 및 unique_id 표시
window = tk.Tk()
label = tk.Label(window, text=unique_id)
label.pack()

# unique_id를 사용자의 문서 폴더에 txt 파일로 저장
docs_path = os.path.join(os.path.expanduser('~'), 'Documents')
with open(os.path.join(docs_path, 'unique_id.txt'), 'w') as f:
    f.write(unique_id)

window.mainloop()
