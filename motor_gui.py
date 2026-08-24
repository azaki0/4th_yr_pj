import serial
import tkinter as tk
from tkinter import Button, Label

class MotorControl:
    def __init__(self, root):
        self.root = root
        self.root.title("3 Motor Control")
        self.root.configure(bg="#000000")

        self.serial_port = None
        self.serial_connected = False
        self.degrees = [80.0, 0.0, 0.0]
        self.ids = ['t', 'q', 'w']
        self.servo_limit = 180#0 top 90 bottom
        self.labels = [None] * 3

        self.presets = {
            "Hello": [0, 0, 0],
            "Wave": [45, 0, 0],
            "Home": [80, 0, 0],
        }

        self.connect_serial()

        for i in range(3):
            Label(self.root, text=f"Motor {i+1} ({self.ids[i]})",
                  bg="#000000", fg="#ffffff", font=("Arial", 14)).grid(row=0, column=i, padx=10, pady=5)
            self.labels[i] = Label(self.root, text=f"{int(self.degrees[i])}",
                                   bg="#000000", fg="#ffffff", font=("Arial", 14))
            self.labels[i].grid(row=1, column=i, padx=10, pady=5)

            up = Button(self.root, text="+", bg="#3C3645", fg="#ffffff",
                        activebackground="#3E3B3B", width=10,
                        command=lambda idx=i: self.change(idx, +1))
            up.grid(row=2, column=i, padx=10, pady=5)

            down = Button(self.root, text="-", bg="#3C3645", fg="#ffffff",
                          activebackground="#3E3B3B", width=10,
                          command=lambda idx=i: self.change(idx, -1))
            down.grid(row=3, column=i, padx=10, pady=5)

        for name in self.presets:
            Button(self.root, text=name, bg="#3C3645", fg="#ffffff",
                   activebackground="#3E3B3B", width=10,
                   command=lambda n=name: self.play_preset(n)).grid(row=5, column=list(self.presets).index(name), padx=10, pady=5)

    def connect_serial(self):
        try:
            self.serial_port = serial.Serial('COM3', 9600)
            self.serial_connected = True
            print("Connected to serial port COM3")
        except:
            print("Error! Could not connect to serial port COM3")
            self.serial_connected = False

    def clamp(self, index, value):
        if index == 0:
            return max(0, min(self.servo_limit, value))
        return max(0, value)

    def change(self, index, direction):
        v = 2.0
        self.degrees[index] = self.clamp(index, self.degrees[index] + v * direction)
        self.labels[index].config(text=f"{int(self.degrees[index])}")
        self.send_all()

    def play_preset(self, name):
        targets = self.presets[name]
        for idx in range(3):
            self.degrees[idx] = self.clamp(idx, targets[idx])
            self.labels[idx].config(text=f"{int(self.degrees[idx])}")
        self.send_all()

    def send_all(self):
        msg = ",".join(f"{self.ids[i]}:{int(self.degrees[i])}" for i in range(3))
        if self.serial_connected:
            try:
                self.serial_port.write(msg.encode())
                print(msg)
            except:
                print("Error sending serial!")

if __name__ == "__main__":
    root = tk.Tk()
    app = MotorControl(root)
    root.mainloop()