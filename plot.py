import re
import json
import itertools
import matplotlib.dates as mdates

from pprint import pprint
from datetime import datetime, timedelta
from matplotlib import pyplot as plt


LATENCY_FILE = "data/latency/2024-11-08/ping-10ms-2m-2024-11-08-02-36-41.txt"
THROUGHPUT_FILE = "data/latency/2024-11-08/iperf3-2m-2024-11-08-02-36-41.json"


def duplicate_object_pairs_hook(pairs):
    def _key(pair):
        (k, v) = pair
        return k
    def gpairs():
        for (k, group) in itertools.groupby(pairs, _key):
            ll = [v for (_, v) in group]
            (v, *extra) = ll
            yield (k, ll if extra else v)
    return dict(gpairs())


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



def load_iperf3(filename: str):
    try:
        with open(filename, "r") as f:
            data = json.loads(f.read(), object_pairs_hook=duplicate_object_pairs_hook)
            recv_mbps = []
            iperf3_timestamps = []
            start_ts = float(data["start"]["timestamp"]["timesecs"])

            for round in data["intervals"]:
                sum = round["sum"]
                iperf3_timestamps.append(datetime.fromtimestamp(start_ts + sum["end"]))
                recv_mbps.append(float(sum["bits_per_second"])/1000000)

    except OSError:
        print("FileNotFoundError: ", filename)
        return

    return iperf3_timestamps, recv_mbps


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

    iperf3_timestamps, recv_mbps = load_iperf3(THROUGHPUT_FILE)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize =(20, 6), sharex=True)

    ax1.plot(timestamps, rtt_list, '.', markersize=1, linestyle='None')
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.set_ylabel("RTT (ms)")

    satellites = load_satellite_info()
    min_timestamp = min(timestamps)
    max_timestamp = max(timestamps)
    for timestamp, satellite in satellites.items():
        timestamp = timestamp.replace(tzinfo=None)
        if timestamp < min_timestamp:
            continue
        elif timestamp > max_timestamp:
            continue
        ax1.text(timestamp, 60, satellite[0], rotation=0)
        distance = f"{satellite[1]:.2f}"
        ax1.text(timestamp, 80, distance, rotation=0)

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
            ax1.axvline(t, color='r', linestyle='--')
            ax2.axvline(t, color='r', linestyle='--')

    ax2.plot(iperf3_timestamps, recv_mbps)
    ax2.set_xlim(min(min(timestamps), min(iperf3_timestamps)), max(max(timestamps), max(iperf3_timestamps)))
    ax2.set_ylim(-1, max(recv_mbps)*1.1)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Throughput (Mbps)")
    ax2.set_title("Downlink Throughput (UDP)")

    plt.tight_layout()
    plt.savefig('ping.png')


if __name__ == "__main__":
    read_ping_latency(LATENCY_FILE)