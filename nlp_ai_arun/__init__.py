import logging
import pandas as pd
import azure.functions as func
import logging
from azure.storage.blob import BlobServiceClient
import os
import pyodbc
    
def main(myblob: func.InputStream):
    
    logging.info(f"Blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes")  
    template_file_path = "https://arunakcs.blob.core.windows.net/excelfiles/main_template/test_template.xlsx"
    df_template = pd.read_excel(template_file_path)
    logging.info(df_template.columns)
    
    
    server = "akcserver.database.windows.net"
    database = "dbarunsql"
    username = "Arun"
    password = "Asds@2022"
    table_name = "TransactionDetails"

    try:
        # Establish the database connection
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
        conn = pyodbc.connect(conn_str)

        # Insert blob details into the SQL table
        cursor = conn.cursor()
        # cursor.execute(f"INSERT INTO {table_name} (FileName, BlobSize) VALUES (?, ?)", name, blob.length)
        # conn.commit()

        # Read rows from the SQL table
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        # Process the retrieved data
        result = []
        for row in rows:
            result.append(list(row))

        logging.info(result)

    except Exception as e:
        logging.error(e)
        raise
    finally:
        conn.close()
    
    