import logging
import pandas as pd
import azure.functions as func
import logging
from azure.storage.blob import BlobServiceClient, ContentSettings
import os
import pyodbc
import re
from datetime import datetime
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer

def read_user_input_data(input_file, df_input):
    current_datetime = datetime.now()
    Current_Date= current_datetime.strftime("%Y-%m-%d")
    filter_date = pd.to_datetime(Current_Date).date()
    
    logging.info(filter_date)
    
    df_input['ModelDate'] = pd.to_datetime(df_input['ModelCopyDateTime'])
    df_input['ModelDateExtracted']=df_input['ModelDate'].dt.date
    df_input =  df_input[df_input['ModelDateExtracted']==filter_date]
    
    if 'Emirates-NBD-Classic-Luxury-Main' in input_file:
        nbd_df = df_input[df_input['DomainName']=='Emirates NBD-Classic Luxury-Main']  
        return nbd_df
    
    elif 'Rak-Bank-Classic-Luxury' in input_file:
        clt_rak_df = df_input[df_input['DomainName']=='Rak Bank-Classic Luxury']
        return clt_rak_df
    
    elif 'CLT-ADCB' in input_file:
        cbd_df = df_input[df_input['DomainName']=='CLT - ADCB']
        return cbd_df
    
    elif 'CBD-Bank' in input_file:
        cbd_df = df_input[df_input['DomainName']=='CBD Bank']
        return cbd_df
    
    elif 'EIB-Loan account' in input_file:
        cbd_df = df_input[df_input['DomainName']=='EIB - Loan account']
        return cbd_df
    
    elif 'OLT-Emirates Islamic Bank' in input_file:
        cbd_df = df_input[df_input['DomainName']=='OLT - Emirates Islamic Bank']
        return cbd_df
    
    elif 'Emirates-NBD-Classic-Passenger' in input_file:
        cbd_df = df_input[df_input['DomainName']=='Emirates NBD-Classic Passenger']
        return cbd_df
    
    elif 'ENBD-Classic-Riders' in input_file:
        cbd_df = df_input[df_input['DomainName']=='ENBD - Classic Riders']
        return cbd_df
    
    else: 
        print(f'{input_file} Does not exist.')
        
def general_preprocess(input_df):
    dropped_df = input_df['TransactionDate'].dropna()
    return input_df

def preprocess_template_data(input_dataframe):
    lowered_description = input_dataframe['Description'].apply(str.lower)
    input_dataframe['description_lowered'] = lowered_description
    return input_dataframe

def preprocess_text_data(input_df):
    lowered_narration_data_series = input_df['Narration'].apply(lambda x: str(x).lower() if x is not None else None)
    input_df['lowered_narration'] = lowered_narration_data_series
    return input_df

def predict_transactions(new_transactions, model_path, vectorizer_path):
    # Load the saved model
    model = load(model_path)

    # Load the saved vectorizer
    vectorizer = load(vectorizer_path)

    # Vectorize the new data
    new_transactions_vectorized = vectorizer.transform(new_transactions)

    # Predict using the loaded model
    predictions = model.predict(new_transactions_vectorized)

    return predictions

def fetch_data_from_sql(server, database, username, password, table_name):
    # Establish the database connection
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
    conn = pyodbc.connect(conn_str)

    # Read rows from the SQL table
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()

    # Convert the result to a DataFrame
    df =pd.DataFrame([tuple(row) for row in rows], columns=[column[0] for column in cursor.description])
    conn.close()
    return df

def populate_final_report(report_template, nlp_classified_df, input_file_path):
    if 'Emirates-NBD-Classic-Luxury-Main' in input_file_path:
        
        for description in report_template['Description']:
            filtered_df = nlp_classified_df[nlp_classified_df['Prediction'] == description]
            debit_sum = filtered_df['Debit'].sum()
            credit_sum = filtered_df['Credit'].sum()
            total_sum = credit_sum + (-debit_sum)
            report_template.loc[report_template['Description'] == description, 'Emirates NBD-Classic Luxury-Main'] = total_sum
        report_template.loc[report_template['Description'] == 'Closing Balance at the day end', 'Emirates NBD-Classic Luxury-Main'] = report_template['Emirates NBD-Classic Luxury-Main'][1:12].sum()
        
    elif 'CBD-Bank' in input_file_path:
        
        for description in report_template['Description']:
            filtered_df = nlp_classified_df[nlp_classified_df['Prediction'] == description]
            debit_sum = filtered_df['Debit'].sum()
            credit_sum = filtered_df['Credit'].sum()
            total_sum = credit_sum + (-debit_sum)
            report_template.loc[report_template['Description'] == description, 'CBD Bank'] = total_sum
        report_template.loc[report_template['Description'] == 'Closing Balance at the day end', 'CBD Bank'] = report_template['CBD Bank'][1:12].sum()

    elif 'Rak-Bank-Classic-Luxury' in input_file_path:
        
        for description in report_template['Description']:
            filtered_df = nlp_classified_df[nlp_classified_df['Prediction'] == description]
            debit_sum = filtered_df['Debit'].sum()
            credit_sum = filtered_df['Credit'].sum()
            total_sum = credit_sum + (-debit_sum)
            report_template.loc[report_template['Description'] == description, 'Emirates NBD-Classic Luxury-Main'] = total_sum
        report_template.loc[report_template['Description'] == 'Closing Balance at the day end', 'Emirates NBD-Classic Luxury-Main'] = report_template['Emirates NBD-Classic Luxury-Main'][1:12].sum()

        
        return report_template


def save_dataframe_to_blob(dataframe, connection_string, container_name, file_name):
    # Save DataFrame to Excel file
    excel_file = f"{file_name}.xlsx"
    dataframe.to_excel(excel_file, index=False)

    # Upload Excel file to Blob storage
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(excel_file)

    with open(excel_file, "rb") as file:
        blob_client.upload_blob(file, overwrite=True, content_settings=ContentSettings(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'))

    # Delete the local Excel file
    os.remove(excel_file)

def main(myblob: func.InputStream):
    
    logging.info(f"Blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes")  
    template_file_path = "https://arunakcs.blob.core.windows.net/excelfiles/main_template/test_template.xlsx"
    df_template = pd.read_excel(template_file_path)
    df_reference = pd.read_excel(template_file_path, sheet_name="term_references")
    logging.info(df_template.columns)
    
    model_path = "https://asdsandassociates.sharepoint.com/:u:/s/AIProject/EU5AbNR6OI1GgYqeAAcFw5wBuv42N_mcTG5PapwFzVWxDg?e=63bc1I"
    vectorizer_path = "https://asdsandassociates.sharepoint.com/:u:/s/AIProject/EZ6mxwIESiJLlGCWPB12dNUB3IPFqpA0eVeqhijsUCyMNQ?e=H3RhgT"
    server = "akcserver.database.windows.net"
    database = "dbarunsql"
    username = "Arun"
    password = "Asds@2022"
    table_name = "TransactionDetails"

    try:
        
        df_input_bank_statement_from_sql = fetch_data_from_sql(server, database, username, password, table_name)
        df_input_bank_statement = read_user_input_data(myblob.name,df_input_bank_statement_from_sql)
        preprocessed_data = general_preprocess(df_input_bank_statement)
        pdf_based_file_preprocessed_data = preprocess_text_data(preprocessed_data)
        report_template = preprocess_template_data(df_template)
 
        
        nlp_classified = pdf_based_file_preprocessed_data.dropna(subset=['Narration'])
        nlp_classified_data_without_nulls = nlp_classified.dropna(subset=['Narration'])
        nlp_bank_transactions = nlp_classified_data_without_nulls['Narration']
        
        predictions = predict_transactions(nlp_bank_transactions, model_path, vectorizer_path)
        
        # Print the predictions
        for transaction, prediction in zip(nlp_bank_transactions, predictions):
            logging.info(f'Transaction: {transaction}\nPrediction: {prediction}')

        pdf_based_file_preprocessed_data['Prediction'] = predictions
        populate_report_template = populate_final_report(report_template, pdf_based_file_preprocessed_data, myblob.name)
        
        now_date = datetime.now()
        connection_string = "DefaultEndpointsProtocol=https;AccountName=arunakcs;AccountKey=nx8T5960W1vcaeHKOD/4HtiCm0/n58VXhtsNAp7LoyDdZX6IdRPsomJsBoOgB72wPd9AHfwwcoFo+AStndZq2Q==;EndpointSuffix=core.windows.net"
        container_name = "outputreport"
        file_name = f"output_report_{now_date}"
        
        save_dataframe_to_blob(populate_report_template,connection_string, container_name, file_name)

    except Exception as e:
        logging.error(e)
    
