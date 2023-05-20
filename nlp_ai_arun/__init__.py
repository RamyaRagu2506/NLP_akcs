import logging
import pandas as pd
import azure.functions as func
# from azure.storage.blob import BlobServiceClient
# import io
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
        # # Connect to the Azure Blob storage using the connection string
        # connection_string = "DefaultEndpointsProtocol=https;AccountName=arunakcs;AccountKey=nx8T5960W1vcaeHKOD/4HtiCm0/n58VXhtsNAp7LoyDdZX6IdRPsomJsBoOgB72wPd9AHfwwcoFo+AStndZq2Q==;EndpointSuffix=core.windows.net"
        # blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        # # Get the blob container and blob client
        # container_name = "excelfiles"
        # container_client = blob_service_client.get_container_client(container_name)
        # blob_client = container_client.get_blob_client(myblob.name)

        # excel_data = blob_client.download_blob().readall()
        df = pd.read_excel(excel_complete_path)
        logging.df(df.head())
        
    except Exception as e:
        logging.error(f"Error processing blob: {e}")
    
    