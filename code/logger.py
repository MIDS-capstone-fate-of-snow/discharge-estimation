"""Set up application logging configuration."""

import datetime
import logging
import os

import pytz

# Get this directory location:
DIR, FILENAME = os.path.split(__file__)
LOGDIR = os.path.join(os.path.dirname(DIR), "logs")
if not os.path.exists(LOGDIR):
    os.mkdir(LOGDIR)

utc_now = pytz.utc.localize(datetime.datetime.utcnow())
time_stamp = utc_now.astimezone(pytz.timezone("US/Eastern")).strftime("%Y_%m_%d %H;%M;%S")
logpath = os.path.join(LOGDIR, f"discharge_estimation_log_{time_stamp}.log")
with open(logpath, "w") as f:
    pass
log_format = "[%(asctime)-15s] [%(levelname)08s] [%(funcName)s] %(message)s [line %(lineno)d]"
logging.basicConfig(level=logging.DEBUG, filename=logpath, format=log_format)
logger = logging.getLogger("discharge_estimation")
