import logging
import pandas as pd
import azure.functions as func
from azure.storage.blob import BlobServiceClient



def main(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes")
    try:
        
        # Connect to the Azure Blob storage using the connection string
        connection_string = "DefaultEndpointsProtocol=https;AccountName=arunakcs;AccountKey=nx8T5960W1vcaeHKOD/4HtiCm0/n58VXhtsNAp7LoyDdZX6IdRPsomJsBoOgB72wPd9AHfwwcoFo+AStndZq2Q==;EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        # Get the blob container and blob client
        container_name = "excelfiles"
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(myblob.name)
    
        df = pd.read_excel(myblob.name)
        logging.df(df.head())
        
    except Exception as e:
        logging.error(f"Error processing blob: {e}")
    
    