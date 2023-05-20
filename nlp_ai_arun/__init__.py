import logging
import pandas as pd
import azure.functions as func
import logging

logging.basicConfig(level=logging.DEBUG)


def main(NBDblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {NBDblob.name} \n"
                 f"Blob Size: {NBDblob.length} bytes")
    
    try:
        
        account_name = "https://arunakcs.blob.core.windows.net/"
        excel_complete_path = account_name + NBDblob.name
        logging.info(excel_complete_path)
        df = pd.read_excel(excel_complete_path)
        logging.info(df.head())
        
    except Exception as e:
        logging.error(f"Error processing blob: {e}")
    
    