import socket
from datetime import datetime

def write_results_to_csv(results, basename, header):
    hostname = socket.gethostname().split('.')[0]
    today = datetime.now().strftime('%Y%m%d')
    filename = f"{basename}_{hostname}_{today}.csv"
    with open(filename, "w") as f:
        f.write(header + "\n")
        for line in results:
            f.write(line + "\n")