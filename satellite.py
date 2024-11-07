#!/usr/bin/env python3
import os
import csv
import logging
import sys
import time
import math
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


_tilt = None
_rotation_az = None
_lat = None
_lon = None
_alt = None
satellites = None

logger = logging.getLogger(__name__)
current_timeslot_start = None


def set_observation_time(year, month, day, hour, minute, second):
    ts = load.timescale()
    return ts.utc(year, month, day, hour, minute, second)


def preprocess_observed_data(filename):
    data = pd.read_csv(filename, sep=',', header=None, names=['Timestamp', 'Y', 'X'])
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], utc=True)

    observer_x, observer_y = 62, 62  # Assume this is the observer's pixel location
    pixel_to_degrees = (80/62)  # Conversion factor from pixel to degrees

    positions = []
    for index, point in data.iterrows():
        dx, dy = point['X'] - observer_x, (123 - point['Y']) - observer_y
        radius = np.sqrt(dx**2 + dy**2) * pixel_to_degrees
        azimuth = np.degrees(np.arctan2(dx, dy))
        # Normalize the azimuth to ensure it's within 0 to 360 degrees
        azimuth = (azimuth + 360) % 360
        elevation = 90 - radius
        positions.append((point['Timestamp'], point['Y'], point['X'], elevation, azimuth))

    df_positions = pd.DataFrame(positions, columns=['Timestamp', 'Y', 'X', 'Elevation', 'Azimuth'])
    return df_positions


def process_observed_data(start_time, merged_data_df):
    # data = pd.read_csv(filename, sep=',', header=None, names=['Timestamp', 'Y', 'X'])
    # data['Timestamp'] = pd.to_datetime(data['Timestamp'], utc=True)
    interval_start_time = pd.to_datetime(start_time, utc=True)
    interval_end_time = interval_start_time + pd.Timedelta(seconds=14)
    # filtered_data = data[(data['Timestamp'] >= interval_start_time) & (data['Timestamp'] < interval_end_time)]
    # if filtered_data.empty:
    #     print("No data found.")
    #     return None

    # merged_data_df['Timestamp'] = pd.to_datetime(merged_data_df['Timestamp'], utc=True)
    # merged_filtered_data = merged_data_df[(merged_data_df['Timestamp'] >= interval_start_time) & (merged_data_df['Timestamp'] < interval_end_time)]
    merged_filtered_data = merged_data_df

    if merged_filtered_data.empty:
        print("No matching data found in merged_data_file.")
        return None

    if len(merged_filtered_data) < 3:
        print("Not enough data points in merged_filtered_data.")
        return None

    start_data = merged_filtered_data.iloc[0]
    middle_data = merged_filtered_data.iloc[len(merged_filtered_data)//2]
    end_data = merged_filtered_data.iloc[-2]
    rotation = 0
    positions = [
        (start_data['Timestamp'], (90 - start_data['Elevation'], (start_data['Azimuth'] + rotation) % 360)),
        (middle_data['Timestamp'], (90 - middle_data['Elevation'], (middle_data['Azimuth'] + rotation) % 360)),
        (end_data['Timestamp'], (90 - end_data['Elevation'], (end_data['Azimuth'] + rotation) % 360))
    ]

    return positions

# Calculate angular separation between two positions
def angular_separation(alt1, az1, alt2, az2):
    """Calculate the angular separation between two points on a sphere given by altitude and azimuth."""
    alt1, alt2 = np.radians(alt1), np.radians(alt2)
    az1 = (az1 + 360) % 360
    az2 = (az2 + 360) % 360
    az_diff = np.abs(az1 - az2)
    if az_diff > 180:
        az_diff = 360 - az_diff
    az_diff = np.radians(az_diff)
    separation = np.arccos(np.sin(alt1) * np.sin(alt2) + np.cos(alt1) * np.cos(alt2) * np.cos(az_diff))
    return np.degrees(separation)

# Calculate bearing (direction) between two points
def calculate_bearing(alt1, az1, alt2, az2):
    alt1, alt2 = np.radians(alt1), np.radians(alt2)
    az1, az2 = np.radians(az1), np.radians(az2)
    x = np.sin(az2 - az1) * np.cos(alt2)
    y = np.cos(alt1) * np.sin(alt2) - np.sin(alt1) * np.cos(alt2) * np.cos(az2 - az1)
    bearing = np.arctan2(x, y)
    bearing = np.degrees(bearing)
    return (bearing + 360) % 360

# Calculate bearing difference between two trajectories
def calculate_bearing_difference(observed_trajectory, satellite_trajectory):
    observed_bearing = calculate_bearing(observed_trajectory[0][0], observed_trajectory[0][1],
                                         observed_trajectory[-1][0], observed_trajectory[-1][1])
    satellite_bearing = calculate_bearing(satellite_trajectory[0][0], satellite_trajectory[0][1],
                                          satellite_trajectory[-1][0], satellite_trajectory[-1][1])
    bearing_diff = abs(observed_bearing - satellite_bearing)
    if bearing_diff > 180:
        bearing_diff = 360 - bearing_diff
    return bearing_diff

# Calculate the total angular separation and bearing difference
def calculate_total_difference(observed_positions, satellite_positions):
    total_angular_separation = 0
    for i in range(len(observed_positions)):
        obs_alt, obs_az = observed_positions[i]
        sat_alt, sat_az = satellite_positions[i]
        separation = angular_separation(obs_alt, obs_az, sat_alt, sat_az)
        total_angular_separation += separation
    bearing_diff = calculate_bearing_difference(observed_positions, satellite_positions)
    total_difference = total_angular_separation + bearing_diff
    return total_difference

# Update the find_matching_satellites function to use angular separation and bearing difference
def find_matching_satellites(satellites, observer_location, observed_positions_with_timestamps):
    best_match = None
    closest_total_difference = float('inf')

    ts = load.timescale()

    for satellite in satellites:
        satellite_positions = []
        valid_positions = True

        for observed_time, observed_data in observed_positions_with_timestamps:
            difference = satellite - observer_location
            topocentric = difference.at(ts.utc(observed_time.year, observed_time.month, observed_time.day, observed_time.hour, observed_time.minute, observed_time.second))
            alt, az, _ = topocentric.altaz()

            if alt.degrees <= 20:
                valid_positions = False
                break

            satellite_positions.append((alt.degrees, az.degrees))

        if valid_positions:
            total_difference = calculate_total_difference(
                [(90 - data[0], data[1]) for _, data in observed_positions_with_timestamps],
                satellite_positions
            )
            if total_difference < closest_total_difference:
                closest_total_difference = total_difference
                best_match = satellite.name

    return [best_match] if best_match else []

def calculate_distance_for_best_match(satellite, observer_location, start_time, interval_seconds):
    ts = load.timescale()
    distances = []
    for second in range(0, interval_seconds + 1):
        current_time = start_time + timedelta(seconds=second)
        difference = satellite - observer_location
        topocentric = difference.at(current_time)
        distance = topocentric.distance().km
        distances.append(distance)
    return distances


def estimate(year, month, day, hour, minute, second, merged_data_df, satellites):
    get_dish_status()

    initial_time = set_observation_time(year, month, day, hour, minute, second)
    observer_location =wgs84.latlon(latitude_degrees=_lat, longitude_degrees= _lon, elevation_m=_alt)
    interval_seconds = 15
    observed_positions_with_timestamps = process_observed_data(initial_time.utc_strftime('%Y-%m-%dT%H:%M:%SZ'), merged_data_df)
    if observed_positions_with_timestamps is None:
        return [], [], []

    matching_satellites = find_matching_satellites(satellites, observer_location, observed_positions_with_timestamps)
    if not matching_satellites:
        return observed_positions_with_timestamps, [], []

    best_match_satellite = next(sat for sat in satellites if sat.name == matching_satellites[0])
    distances = calculate_distance_for_best_match(best_match_satellite, observer_location, initial_time, 14)

    return observed_positions_with_timestamps, matching_satellites, distances

def process_intervals(start_year, start_month, start_day, start_hour, start_minute, start_second, end_year, end_month, end_day, end_hour, end_minute, end_second, merged_data_df, satellites):
    results = []

    start_time = datetime(start_year, start_month, start_day, start_hour, start_minute, start_second, tzinfo=utc)
    end_time = datetime(end_year, end_month, end_day, end_hour, end_minute, end_second, tzinfo=utc)
    current_time = start_time

    while current_time <= end_time:
        print(f"Processing data for {current_time}")
        observed_positions_with_timestamps, matching_satellites, distances = estimate(current_time.year, current_time.month, current_time.day, current_time.hour, current_time.minute, current_time.second, merged_data_df, satellites)
        if matching_satellites:
            for second in range(15):
                if second < len(distances):
                    results.append({
                        'Timestamp': current_time + timedelta(seconds=second),
                        'Connected_Satellite': matching_satellites[0],
                        'Distance': distances[second]
                    })
        current_time += timedelta(seconds=15)

    result_df = pd.DataFrame(results)
    return result_df


def get_dish_status():
    global _lat
    global _lon
    global _alt
    global _tilt
    global _rotation_az

    context = starlink_grpc.ChannelContext(target=STARLINK_GRPC_ADDR_PORT)
    location = starlink_grpc.get_location(context)
    if "lla" in location:
        # print("Dish location:", location)
        _lat = location.lla.lat
        _lon = location.lla.lon
        _alt = location.lla.alt
    else:
        print("Get dish location failed")
        exit(1)

    dish_status = starlink_grpc.get_status(context)
    # print("Dish alignment status:", dish_status.alignment_stats)
    _tilt = dish_status.alignment_stats.tilt_angle_deg
    _rotation_az = dish_status.alignment_stats.boresight_azimuth_deg
    if not _tilt or not _rotation_az:
        print("Get dish alignment stats failed")
        exit(1)


def get_connected_satellite(snapshots):
    start_time_dt = datetime.strptime(current_timeslot_start, "%Y-%m-%d %H:%M:%S")
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


def capture_snr_data(duration_seconds, interval_seconds, context):
    snapshots = []
    end_time = time.time() + duration_seconds

    previous_snr_data = np.zeros_like(snapshots[0][1])
    while time.time() < end_time:
        try:
            snr_data = starlink_grpc.obstruction_map(context)
            snr_data_array = np.array(snr_data, dtype=int)
            snr_data_array[snr_data_array == -1] = 0
            snapshots.append(snr_data_array)

            get_connected_satellite(previous_snr_data, snapshots)

            previous_snr_data = snr_data
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
    global current_timeslot_start

    target_seconds = {12, 27, 42, 57}
    while True:
        current_second = datetime.now(timezone.utc).second
        if current_second in target_seconds:
            current_timeslot_start = datetime.now(timezone.utc)
            break
        time.sleep(0.1)


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
