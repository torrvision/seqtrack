import datetime
import errno
import os

def get_time():
    dt = datetime.datetime.now()
    time = '{0:04d}{1:02d}{2:02d}_h{3:02d}m{4:02d}s{5:02d}'.format(
            dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second)
    return time

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

