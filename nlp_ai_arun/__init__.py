import logging
import pandas as pd
import azure.functions as func
import logging
import re
import datetime
from datetime import datetime as second_datetime
from azure.storage.blob import BlobServiceClient
import io
import requests

logging.basicConfig(level=logging.DEBUG)
    
def main(NBDblob: func.InputStream):
    
    logging.info(NBDblob.name,"Checking if the trigger works")    
    