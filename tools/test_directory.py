import os
import glob

log_dir = os.path.join(os.getcwd(), "logs")
print(f"Looking in: {log_dir}")

sessions = glob.glob(os.path.join(log_dir, "session_*"))
print(f"Found session folders: {sessions}")