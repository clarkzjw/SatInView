#!/usr/bin/env python3

import csv
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from skyfield.api import load, wgs84, utc

from config import STARLINK_GRPC_ADDR_PORT, TLE_URL, TLE_DATA_DIR, DURATION_SECONDS, DISH_ID
from util import date_time_string, ensure_directory, ensure_data_directory
from fov import estimate

sys.path.insert(0,str(Path('./starlink-grpc-tools').resolve()))
import starlink_grpc


satellites = None

logger = logging.getLogger(__name__)


def capture_snr_data(duration_seconds, interval_seconds, context):
    snapshots = []
    end_time = time.time() + duration_seconds

    while time.time() < end_time:
        try:
            snr_data = starlink_grpc.obstruction_map(context)
            snr_data_array = np.array(snr_data, dtype=int)
            snr_data_array[snr_data_array == -1] = 0
            snapshots.append(snr_data_array)
            time.sleep(interval_seconds)
        except starlink_grpc.GrpcError as e:
            print("Failed getting obstruction map data:", str(e))
            break

    return snapshots


def save_white_pixel_coordinates_xor(directory, filename, snapshots, start_time):
    start_time_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    previous_snr_data = np.zeros_like(snapshots[0][1])

    observer_x, observer_y = 62, 62  # Assume this is the observer's pixel location
    pixel_to_degrees = (80/62)  # Conversion factor from pixel to degrees

    merged_data_df = pd.DataFrame(columns=['Timestamp', 'Y', 'X', 'Elevation', 'Azimuth'])
    results = []

    with open("{}/obstruction-data-{}-{}.csv".format(directory, DISH_ID, filename), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        i = 0
        hold_coord = None  # Initialize as None

        for snr_data in snapshots:
            xor_snr_data = np.bitwise_xor(previous_snr_data, snr_data)
            coords = np.argwhere(xor_snr_data == 1)

            if coords.size > 0:
                coord = coords[-1]  # Get the last occurrence
                hold_coord = coord  # Update hold_coord
            elif hold_coord is not None:
                coord = hold_coord  # Use the previous hold_coord if coords is empty
            else:
                continue  # If both coords is empty and hold_coord is None, skip this iteration

            # white_pixel_coords.append((start_time_dt + timedelta(seconds=i), tuple(coord)))
            previous_snr_data = snr_data
            i += 1

            Y = coord[0]
            X = coord[1]

            dx, dy = X - observer_x, (123 - Y) - observer_y
            radius = np.sqrt(dx**2 + dy**2) * pixel_to_degrees
            azimuth = np.degrees(np.arctan2(dx, dy))
            # Normalize the azimuth to ensure it's within 0 to 360 degrees
            azimuth = (azimuth + 360) % 360
            elevation = 90 - radius

            timestamp = pd.to_datetime(start_time_dt + timedelta(seconds=i), utc=True)
            record = {
                "Timestamp": timestamp,
                "Y": Y,
                "X": X,
                "Elevation": elevation,
                "Azimuth": azimuth
            }
            if merged_data_df.empty:
                # Initialize the DataFrame with the new record if it's empty
                merged_data_df = pd.DataFrame([record])
            else:
                # Otherwise, append the new record to the existing DataFrame
                merged_data_df = pd.concat([merged_data_df, pd.DataFrame([record])], ignore_index=True)
            # merged_data_df = pd.concat([merged_data_df, pd.DataFrame([record])], ignore_index=True)
            writer.writerow([timestamp, Y, X, elevation, azimuth])

            current_time = datetime.now(timezone.utc)
            print(f"Processing data for {current_time}")
            observed_positions_with_timestamps, matching_satellites, distances = estimate(current_time.year, current_time.month, current_time.day, current_time.hour, current_time.minute, current_time.second, merged_data_df, satellites)
            if matching_satellites:
                for second in range(15):
                    if second < len(distances):
                        record = {
                            'Timestamp': current_time + timedelta(seconds=second),
                            'Connected_Satellite': matching_satellites[0],
                            'Distance': distances[second]
                        }
                        results.append(record)
                        print(record)



def wait_until_target_time():
    target_seconds = {12, 27, 42, 57}
    while True:
        current_second = datetime.now(timezone.utc).second
        if current_second in target_seconds:
            break
        time.sleep(0.5)


def load_satellites():
    global satellites
    directory = Path(TLE_DATA_DIR).joinpath(ensure_data_directory(TLE_DATA_DIR))
    satellites = load.tle_file(TLE_URL, True, "{}/starlink-tle-{}.txt".format(directory, date_time_string()))
    logging.info("Loaded {} Starlink satellites".format(len(satellites)))
    return satellites


def collect_obstruction_data():
    load_satellites()

    start = datetime.now(timezone.utc)
    context = starlink_grpc.ChannelContext(target=STARLINK_GRPC_ADDR_PORT)

    all_snapshots = []
    start_times = []
    end_times = []

    timeslot_duration_seconds = 14
    interval_seconds = 1  # Capture a snapshot every 1 second

    round = 0

    directory = Path(TLE_DATA_DIR).joinpath(ensure_data_directory(TLE_DATA_DIR))
    ensure_directory(str(directory))
    filename = date_time_string()

    while True:
        now = datetime.now(timezone.utc)
        if now - start >= timedelta(seconds=DURATION_SECONDS):
            return

        round += 1
        logging.info("Current round {}".format(round))

        wait_until_target_time()

        starlink_grpc.reset_obstruction_map(context)

        start_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        snapshots = capture_snr_data(timeslot_duration_seconds, interval_seconds, context)

        end_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        all_snapshots.append(snapshots)
        start_times.append(start_time)
        end_times.append(end_time)

        save_white_pixel_coordinates_xor(directory, filename, snapshots, start_time)
