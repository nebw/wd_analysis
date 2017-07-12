from datetime import datetime
import time
import os
import numpy as np
import pandas as pd
import pytz
from astropy.coordinates import EarthLocation, AltAz
from astropy.coordinates import get_sun
import astropy.units as u
from astropy.time import Time

from wda.meta import Feeder


def represents_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_numeric_prefix(fname):
    prefix = ''
    for c in fname:
        if represents_integer(c):
            prefix += c
        else:
            break
    return int(prefix)


def extract_datetime(row, timezone=pytz.timezone('Europe/Berlin')):
    if type(row.Tag) == datetime:
        day = row.Tag.day
        month = row.Tag.month
    else:
        day, month = row.Tag.split('.')[:2]
    day = int(day)
    month = int(month)
    year = 2015
    time = row['Uhrzeit\n(Ende)']

    return timezone.localize(
        pd.datetime(year, month, day, time.hour, time.minute, time.second))


def calculate_datetime_start(row):
    return row.datetime_end - pd.Timedelta(row['Länge'], unit='m')


def parse_subfolder(subfolder):
    if type(subfolder) == str:
        return subfolder.split('_')[0]
    elif type(subfolder) == int:
        return subfolder


def extract_azimuth(row, latitude=50.794916, longitude=8.919806):
    time = Time(row.dance_start_time.astimezone(pytz.UTC))
    earth_loc = EarthLocation(lat=latitude*u.deg, lon=longitude*u.deg, height=0*u.m)
    sun_loc = get_sun(time)
    azimuth = -sun_loc.transform_to(AltAz(obstime=time, location=earth_loc)).az

    return azimuth.deg


def get_suncalc(base_path):
    suncalc_path = os.path.join(base_path, 'data', 'zeitpunkt_videos_fuer_suncalc_lot.xlsx')
    suncalc_df = pd.read_excel(suncalc_path)
    suncalc_df['lot_rad'] = (pd.to_numeric(suncalc_df.LOTKORREKTUR,
                                           errors='coerce') / 360) * 2 * np.pi
    del suncalc_df['Azimuth']
    return suncalc_df


def parse_feeder(base_path, feeder_idx, min_waggles=5, constant_correction=2.):
    feeder_str = 'feeder{}.csv'.format(feeder_idx)
    result_path = os.path.join(base_path, 'results', feeder_str)
    feeder_pos = Feeder(feeder_idx).position

    data = pd.read_csv(result_path)
    data = data[data.num_waggles >= min_waggles]

    data['feeder_pos'] = [feeder_pos for _ in range(len(data))]

    if feeder_idx in [1, 2, 3]:
        data['video_id'] = data.subfolder.apply(parse_subfolder).astype(np.int)
    else:
        data['video_id'] = data.fname.apply(get_numeric_prefix).astype(np.int)

    suncalc_df = get_suncalc(base_path)

    data = pd.merge(data, suncalc_df, left_on='video_id', right_on='Video')
    data['angle'] = data['angle'] - data.lot_rad - ((2 * np.pi) / 360) * constant_correction

    data['datetime_end'] = data.apply(extract_datetime, axis=1)
    data['datetime_start'] = data.apply(calculate_datetime_start, axis=1)
    data['start_time_offset'] = data.start_time.apply(lambda t: pd.Timedelta(t, unit='s'))
    data['dance_start_time'] = data['datetime_start'] + data['start_time_offset']
    data['azimuth'] = data.apply(extract_azimuth, axis=1)
    data['azimuth_rad'] = (data.azimuth / 360) * (2 * np.pi)
    data['relative_angle'] = data.azimuth_rad - data.angle

    return data


def get_dance_vectors(feeder_data, distances):
    return np.stack((
        np.cos(feeder_data.relative_angle),
        np.sin(feeder_data.relative_angle)),
        -1) * np.array(distances)[:, None]


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap
