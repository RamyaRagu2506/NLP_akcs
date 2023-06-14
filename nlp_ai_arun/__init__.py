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

    
def main(myblob: func.InputStream):
    
    logging.info(f"Blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes")  
    