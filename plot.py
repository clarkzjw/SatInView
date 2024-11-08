import re
import matplotlib.dates as mdates

from pprint import pprint
from datetime import datetime, timedelta
from matplotlib import pyplot as plt


def load_satellite_info():
    with open("connected_satellite.csv", "r") as f:
        satellites = {}
        previous = None
        for line in f.readlines():
            timestamp, _, _, _, _, distance, satellite = line.strip("\n").split(",")
            satellite = satellite.replace("STARLINK", "SL")
            timestamp = datetime.fromisoformat(timestamp) - timedelta(hours=8)
            distance = float(distance)
            if previous == None:
                satellites[timestamp] = (satellite, distance)
                previous = satellite
            else:
                if previous != satellite:
                    satellites[timestamp] = (satellite, distance)
                    previous = satellite
    return satellites


def read_ping_latency(filename: str):
    print(filename)
    with open(filename, "r") as f:
        count = 0
        seq_list = []
        rtt_list = []
        timestamps = []

        for line in f.readlines():
            match = re.search(r"\[(\d+\.\d+)\].*icmp_seq=(\d+).*time=(\d+\.?\d*)\s*ms", line)
            if match:
                count += 1
                t = datetime.fromtimestamp(float(match.group(1)))
                seq = int(match.group(2))
                rtt = float(match.group(3))
                seq_list.append(seq)
                rtt_list.append(rtt)
                timestamps.append(t)

    fig = plt.figure(figsize =(20, 6))
    ax = fig.add_subplot(111)

    ax.plot(timestamps, rtt_list, '.', markersize=1, linestyle='None')
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.set_ylabel("RTT (ms)")

    satellites = load_satellite_info()
    for timestamp, satellite in satellites.items():
        ax.text(timestamp, 60, satellite[0], rotation=0)
        distance = f"{satellite[1]:.2f}"
        ax.text(timestamp, 70, distance, rotation=0)

    seconds = [12, 27, 42, 57]
    handovers = []

    for figtimestamps in [timestamps]:
        for timestamp in figtimestamps:
            for s in seconds:
                line_time = timestamp.replace(second=s, microsecond=0)
                if timestamp <= line_time < timestamp + timedelta(minutes=1):
                    handovers.append(line_time)

    for t in sorted(list(set(handovers))):
        if t < max(timestamps):
            ax.axvline(t, color='r', linestyle='--')

    plt.tight_layout()
    plt.savefig('ping.png')


if __name__ == "__main__":
    read_ping_latency("data/latency/2024-11-08/ping-10ms-60m-2024-11-08-00-51-02.txt")