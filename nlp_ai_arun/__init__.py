import logging
import pandas as pd
import azure.functions as func
import logging
import re
from datetime import datetime
from azure.storage.blob import BlobServiceClient
import os
import io


logging.basicConfig(level=logging.DEBUG)

def nlp_algo_comparison(file_path, template_path):
    df_of_pdf_input_file = pd.read_excel(file_path)
    template_df = pd.read_excel(template_path, sheet_name='Summary_test')
    
    logging.info(f"Loading keywords...")
    
    # Assuming you have a DataFrame named 'pdf_df' and the keyword list as 'keyword_finance'
    keyword_finance = ['inward remittance', 'network international', 'sdm deposit']

# Initialize empty lists
    debit_ir = []
    credit_ir = []
    debit_cd = []
    credit_cd = []

# Convert the 'Narration' column to lowercase for case-insensitive matching
    df_of_pdf_input_file['Narration'] = df_of_pdf_input_file['Narration'].str.lower()

# Iterate over the keywords
    for keyword in keyword_finance:
        
    # Create a regular expression pattern for the keyword
        pattern = re.compile(keyword, re.IGNORECASE)
    
    # Check if keyword is 'inward remittance' or 'network international'
        if keyword in ['inward remittance', 'network international']:
        # Filter rows where the keyword pattern matches the 'Narration' column
            keyword_rows = df_of_pdf_input_file[df_of_pdf_input_file['Narration'].str.contains(pattern)]

        # Append debit and credit values to the respective lists
            debit_ir.extend(keyword_rows['Debit'].tolist())
            credit_ir.extend(keyword_rows['Credit'].tolist())
        elif keyword == 'sdm deposit':
        # Filter rows where the keyword pattern matches the 'Narration' column
            keyword_rows = df_of_pdf_input_file[df_of_pdf_input_file['Narration'].str.contains(pattern)]
        
            
        # Append debit and credit values to the respective lists
            debit_cd.extend(keyword_rows['Debit'].tolist())
            credit_cd.extend(keyword_rows['Credit'].tolist())
            
    logging.info("Calculating the debit and credit for EmiratedNBD")
    total_ir = sum(credit_ir) - sum(debit_ir)
    total_cd = sum(credit_cd) - sum(debit_cd)
    
    # Convert the 'Description' column to lowercase for case-insensitive matching
    template_df['Description'] = template_df['Description'].str.lower()

    # Update values using vectorized operations
    template_df.loc[template_df['Description'] == 'inward remittance', 'Emirates NBD-Classic Luxury-Main'] = total_ir
    template_df.loc[template_df['Description'] == 'cash deposited', 'Emirates NBD-Classic Luxury-Main'] = total_cd
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    new_filename = f"NBDEmirates_{current_datetime}.xlsx"
    # Rename the file
    try:
        logging.info("Connecting to storage container")    
        connection_string = "DefaultEndpointsProtocol=https;AccountName=arunakcs;AccountKey=nx8T5960W1vcaeHKOD/4HtiCm0/n58VXhtsNAp7LoyDdZX6IdRPsomJsBoOgB72wPd9AHfwwcoFo+AStndZq2Q==;EndpointSuffix=core.windows.net"
        container_name = "excelfiles"
            # Create a BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        # Get a reference to the container
        container_client = blob_service_client.get_container_client(container_name)
        excel_buffer = io.BytesIO()
        template_df.to_excel(excel_buffer,sheet_name='summary_output')
        excel_buffer.seek(0)
        container_client.upload_blob(name=new_filename, data=excel_buffer)
        logging.info("Opening and loading excel to blob storage")

        
        logging.info(f"DataFrame uploaded successfully to Azure Blob Storage as '{new_filename}'.")
        

    except Exception as e:
        logging.error(f"Error uploading DataFrame to Azure Blob Storage: {str(e)}")
    finally:
        # Delete the temporary Excel file
        os.remove(excel_file)

    
def main(NBDblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {NBDblob.name} \n"
                 f"Blob Size: {NBDblob.length} bytes")
    
    try:
        account_name = "https://arunakcs.blob.core.windows.net/"
        excel_complete_path = account_name + NBDblob.name
        template_path = "https://arunakcs.blob.core.windows.net/excelfiles/main_template/test_template.xlsx"
        logging.info(f"Complete file path is  {excel_complete_path}")
        nlp_algo_comparison(excel_complete_path, template_path)
        
        
    except Exception as e:
        logging.error(f"Error processing blob: {e}")
    
    