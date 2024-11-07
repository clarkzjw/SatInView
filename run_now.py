import logging
import schedule

from latency import icmp_ping
from satellite import collect_obstruction_data
from util import run

logger = logging.getLogger(__name__)

schedule.every(1).hours.at(":00").do(run, icmp_ping).tag("Latency")
schedule.every(1).hours.at(":00").do(run, collect_obstruction_data).tag("TLE")


if __name__ == "__main__":
    schedule.run_all()
