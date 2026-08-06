MOTION_COMMANDS = [
    {
        "id": "wave",
        "label": "Wave",
        "command": "a:90,b:30,c:25",
    },
    {
        "id": "random",
        "label": "Random",
        "command": "a:45,b:120,c:60",
    },
    {
        "id": "handshake",
        "label": "Handshake",
        "command": "a:110,b:70,c:35",
    },
]

def parse_servo_command(command):
    parsed = {}
    for part in command.split(","):
        name, value = part.split(":", 1)
        parsed[name.strip()] = int(value.strip())
    return parsed

def send_motion_command(motion):
    parsed = parse_servo_command(motion["command"])
    print(f"Motion placeholder: {motion['id']} -> {motion['command']} -> {parsed}")

    return {
        "ok": True,
        "id": motion["id"],
        "label": motion["label"],
        "command": motion["command"],
        "parsed": parsed,
        "placeholder": True,
    }
