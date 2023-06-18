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
import pickle
from io import BytesIO

#Global Variables declared
CLOSINGBALANCETABLENAME = os.environ["AkcsCLOSINGBALANCETABLENAME"]
SERVER = os.environ["AkcsServer"]
DATABASE = os.environ["AkcsDatabase"]
USERNAME = os.environ["AkcsDBUsername"] 
PASSWORD = os.environ["AkcsDBPassword"]
TRANSACTIONDETAILSTABLENAME = os.environ["AkcsTransactionDetailsTable"]
CONNECTIONSTRING = os.environ["AkcsStorageConnectionString"]   
CONTAINERNAME = os.environ['AkcsTriggerContainerName']
TEMPLATEFILEPATH = os.environ["AkcsTemplate_file_path"]
OUTPUTREPORTCONTIAINERNAME = os.environ["Akcsoutputreport_container_name"]
DATENOW = datetime.now()

def read_user_input_data(input_file, df_input):
    current_datetime = datetime.now()
    Current_Date= current_datetime.strftime("%Y-%m-%d")
    filter_date = pd.to_datetime(Current_Date).date()
    
    logging.info(filter_date)
    
    df_input['ModelDate'] = pd.to_datetime(df_input['ModelCopyDateTime'])
    df_input['ModelDateExtracted']=df_input['ModelDate'].dt.date
    df_input =  df_input[df_input['ModelDateExtracted']==filter_date]
    logging.info(f"{len(df_input)}")
    
    if 'Emirates-NBD-Classic-Luxury-Main' in input_file:
        nbd_df = df_input[df_input['DomainName']=='Emirates-NBD-Classic-Luxury-Main']  
        logging.info(f"{len(nbd_df)}")
        return nbd_df
    
    elif 'Rak-Bank-Classic-Luxury' in input_file:
        clt_rak_df = df_input[df_input['DomainName']=='Rak-Bank-Classic-Luxury']
        return clt_rak_df
    
    elif 'CLT-ADCB' in input_file:
        clt_adcb_df = df_input[df_input['DomainName']=='CLT-ADCB']
        return clt_adcb_df
    
    elif 'CBD-Bank' in input_file:
        cbd_df = df_input[df_input['DomainName']=='CBD-Bank']
        return cbd_df
    
    elif 'EIB-Loan-account' in input_file:
        cbd_df = df_input[df_input['DomainName']=='EIB-Loan account']
        return cbd_df
    
    elif 'OLT-Emirates-Islamic-Bank' in input_file:
        cbd_df = df_input[df_input['DomainName']=='OLT-Emirates-Islamic-Bank']
        return cbd_df

    
    elif 'Emirates-NBD-Classic-Passenger' in input_file:
        emirates_nbd_classic_passenger_df= df_input[df_input['DomainName']=='Emirates-NBD-Classic-Passenger']
        return emirates_nbd_classic_passenger_df
    
    elif 'ENBD-Classic-Riders' in input_file:
        enbd_classic_riders_df = df_input[df_input['DomainName']=='ENBD-Classic-Riders']
        return enbd_classic_riders_df
    
    else: 
        print(f'{input_file} Does not exist.')
        
def general_preprocess(input_df):
    if input_df['TransactionDate'].isnull().sum() > 0:
        dropped_df = input_df['TransactionDate'].dropna()
        return input_df
    return input_df

def preprocess_template_data(input_dataframe):
    lowered_description = input_dataframe['Description'].apply(str.lower)
    input_dataframe['description_lowered'] = lowered_description
    return input_dataframe

def preprocess_text_data(input_df):
    lowered_narration_data_series = input_df['Narration'].apply(lambda x: str(x).lower() if x is not None else None)
    input_df['lowered_narration'] = lowered_narration_data_series
    return input_df

def predict_transactions(new_transactions, model_weights, vectorizer_weights):

    # Vectorize the new data
    new_transactions_vectorized = vectorizer_weights.transform(new_transactions)

    # Predict using the loaded model
    predictions = model_weights.predict(new_transactions_vectorized)

    return predictions

def fetch_data_from_sql(server, database, username, password, table_name):
    # Establish the database connection
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
    conn = pyodbc.connect(conn_str)

    # Read rows from the SQL table
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name} where Classified = 'No'")
    rows = cursor.fetchall()

    # Convert the result to a DataFrame
    df =pd.DataFrame([tuple(row) for row in rows], columns=[column[0] for column in cursor.description])
    conn.close()
    return df

def populate_final_report(report_template, nlp_classified_df, input_file_path, server, database, username, password):
    
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
    conn = pyodbc.connect(conn_str)
    
    if 'Emirates-NBD-Classic-Luxury-Main' in input_file_path:
        logging.info(f"{len(nlp_classified_df)}")
        logging.info(f"{len(report_template)}")
        logging.info(f"{report_template}")
        for description in report_template['Description']:
            filtered_df = nlp_classified_df[nlp_classified_df['Prediction'] == description]
            logging.info(f"{len(filtered_df)}")
            debit_sum = filtered_df['Debit'].sum()
            credit_sum = filtered_df['Credit'].sum()
            total_sum = credit_sum + (-debit_sum)
            report_template.loc[report_template['Description'] == description, 'Emirates NBD-Classic Luxury-Main'] = total_sum
        closingBalance = report_template['Emirates NBD-Classic Luxury-Main'][1:12].sum()
        report_template.loc[report_template['Description'] == 'Closing Balance at the day end', 'Emirates NBD-Classic Luxury-Main'] = report_template['Emirates NBD-Classic Luxury-Main'][1:12].sum()
        logging.info(f"{report_template['Emirates NBD-Classic Luxury-Main']}")
        cursor = conn.cursor()
        sql = f"INSERT INTO {CLOSINGBALANCETABLENAME} (ClosingBalanceDomainCompany, CreatedDate, ModifiedDate, ClosingBalance) " \
              f"VALUES (?, ?, ?, ?)"

# Prepare the values for the parameters
        params = ('Emirates-NBD-Classic-Luxury-Main', datetime.now(), datetime.now(), closingBalance)

# Execute the SQL statement with the parameters
        cursor.execute(sql, params)
        conn.commit()
        conn.close
        
        return report_template
    elif 'CBD-Bank' in input_file_path:
        
        for description in report_template['Description']:
            filtered_df = nlp_classified_df[nlp_classified_df['Prediction'] == description]
            debit_sum = filtered_df['Debit'].sum()
            credit_sum = filtered_df['Credit'].sum()
            total_sum = credit_sum + (-debit_sum)
            report_template.loc[report_template['Description'] == description, 'CBD Bank'] = total_sum
        closingBalance = report_template['CBD Bank'][1:12].sum()
        report_template.loc[report_template['Description'] == 'Closing Balance at the day end', 'CBD Bank'] = report_template['CBD Bank'][1:12].sum()
        cursor = conn.cursor()
        sql = f"INSERT INTO {CLOSINGBALANCETABLENAME} (ClosingBalanceDomainCompany, CreatedDate, ModifiedDate, ClosingBalance) " \
              f"VALUES (?, ?, ?, ?)"

# Prepare the values for the parameters
        params = ('CBD-Bank', datetime.now(), datetime.now(), closingBalance)

# Execute the SQL statement with the parameters
        cursor.execute(sql, params)
        conn.commit()
        conn.close
        return report_template
        return report_template
    
    
    elif 'Rak-Bank-Classic-Luxury' in input_file_path:
        
        for description in report_template['Description']:
            filtered_df = nlp_classified_df[nlp_classified_df['Prediction'] == description]
            debit_sum = filtered_df['Debit'].sum()
            credit_sum = filtered_df['Credit'].sum()
            total_sum = credit_sum + (-debit_sum)
            report_template.loc[report_template['Description'] == description, 'Rak Bank-Classic Luxury'] = total_sum
        closingBalance = report_template['Rak Bank-Classic Luxury'][1:12].sum()
        report_template.loc[report_template['Description'] == 'Closing Balance at the day end', 'Rak Bank-Classic Luxury'] = report_template['Rak Bank-Classic Luxury'][1:12].sum()
        cursor = conn.cursor()
        sql = f"INSERT INTO {CLOSINGBALANCETABLENAME} (ClosingBalanceDomainCompany, CreatedDate, ModifiedDate, ClosingBalance) " \
              f"VALUES (?, ?, ?, ?)"

# Prepare the values for the parameters
        params = ('Rak-Bank-Classic-Luxury', datetime.now(), datetime.now(), closingBalance)

# Execute the SQL statement with the parameters
        cursor.execute(sql, params)
        conn.commit()
        conn.close
        return report_template
    elif 'CLT-ADCB' in input_file_path:
        
        for description in report_template['Description']:
            filtered_df = nlp_classified_df[nlp_classified_df['Prediction'] == description]
            debit_sum = filtered_df['Debit'].sum()
            credit_sum = filtered_df['Credit'].sum()
            total_sum = credit_sum + (-debit_sum)
            report_template.loc[report_template['Description'] == description, 'CLT-ADCB'] = total_sum
        closingBalance = report_template['CLT-ADCB'][1:12].sum()
        report_template.loc[report_template['Description'] == 'Closing Balance at the day end', 'CLT-ADCB'] = report_template['CLT-ADCB'][1:12].sum()
        cursor = conn.cursor()
        sql = f"INSERT INTO {CLOSINGBALANCETABLENAME} (ClosingBalanceDomainCompany, CreatedDate, ModifiedDate, ClosingBalance) " \
              f"VALUES (?, ?, ?, ?)"

# Prepare the values for the parameters
        params = ('CLT-ADCB', datetime.now(), datetime.now(), closingBalance)

# Execute the SQL statement with the parameters
        cursor.execute(sql, params)
        conn.commit()
        conn.close
        return report_template
    elif 'EIB-Loan-account' in input_file_path:
        
        for description in report_template['Description']:
            filtered_df = nlp_classified_df[nlp_classified_df['Prediction'] == description]
            debit_sum = filtered_df['Debit'].sum()
            credit_sum = filtered_df['Credit'].sum()
            total_sum = credit_sum + (-debit_sum)
            report_template.loc[report_template['Description'] == description, 'EIB-Loan account'] = total_sum
        closingBalance = report_template['EIB-Loan account'][1:12].sum()
        report_template.loc[report_template['Description'] == 'Closing Balance at the day end', 'EIB-Loan account'] = report_template['EIB-Loan account'][1:12].sum()
        cursor = conn.cursor()
        sql = f"INSERT INTO {CLOSINGBALANCETABLENAME} (ClosingBalanceDomainCompany, CreatedDate, ModifiedDate, ClosingBalance) " \
              f"VALUES (?, ?, ?, ?)"

# Prepare the values for the parameters
        params = ('EIB-Loan-account', datetime.now(), datetime.now(), closingBalance)

# Execute the SQL statement with the parameters
        cursor.execute(sql, params)
        conn.commit()
        conn.close
        return report_template
    elif 'OLT-Emirates-Islamic-Bank' in input_file_path:
        
        for description in report_template['Description']:
            filtered_df = nlp_classified_df[nlp_classified_df['Prediction'] == description]
            debit_sum = filtered_df['Debit'].sum()
            credit_sum = filtered_df['Credit'].sum()
            total_sum = credit_sum + (-debit_sum)
            report_template.loc[report_template['Description'] == description, 'OLT - Emirates Islamic Bank'] = total_sum
        closingBalance = report_template['OLT - Emirates Islamic Bank'][1:12].sum()
        report_template.loc[report_template['Description'] == 'Closing Balance at the day end', 'OLT - Emirates Islamic Bank'] = report_template['OLT - Emirates Islamic Bank'][1:12].sum()
        cursor = conn.cursor()
        sql = f"INSERT INTO {CLOSINGBALANCETABLENAME} (ClosingBalanceDomainCompany, CreatedDate, ModifiedDate, ClosingBalance) " \
              f"VALUES (?, ?, ?, ?)"

# Prepare the values for the parameters
        params = ('OLT-Emirates-Islamic-Bank', datetime.now(), datetime.now(), closingBalance)

# Execute the SQL statement with the parameters
        cursor.execute(sql, params)
        conn.commit()
        conn.close
        return report_template
    elif 'Emirates-NBD-Classic-Passenger' in input_file_path:
        for description in report_template['Description']:
            filtered_df = nlp_classified_df[nlp_classified_df['Prediction'] == description]
            debit_sum = filtered_df['Debit'].sum()
            credit_sum = filtered_df['Credit'].sum()
            total_sum = credit_sum + (-debit_sum)
            report_template.loc[report_template['Description'] == description, 'Emirates-NBD-Classic-Passenger'] = total_sum
        closingBalance = report_template['Emirates-NBD-Classic-Passenger'][1:12].sum()
        report_template.loc[report_template['Description'] == 'Closing Balance at the day end', 'Emirates-NBD-Classic-Passenger'] = report_template['Emirates-NBD-Classic-Passenger'][1:12].sum()
        cursor = conn.cursor()
        sql = f"INSERT INTO {CLOSINGBALANCETABLENAME} (ClosingBalanceDomainCompany, CreatedDate, ModifiedDate, ClosingBalance) " \
              f"VALUES (?, ?, ?, ?)"

# Prepare the values for the parameters
        params = ('Emirates-NBD-Classic-Passenger', datetime.now(), datetime.now(), closingBalance)

# Execute the SQL statement with the parameters
        cursor.execute(sql, params)
        conn.commit()
        conn.close
        return report_template
    elif 'ENBD-Classic-Riders' in input_file_path:
        for description in report_template['Description']:
            filtered_df = nlp_classified_df[nlp_classified_df['Prediction'] == description]
            debit_sum = filtered_df['Debit'].sum()
            credit_sum = filtered_df['Credit'].sum()
            total_sum = credit_sum + (-debit_sum)
            report_template.loc[report_template['Description'] == description, 'OENBD - Classic Riders'] = total_sum
        closingBalance = report_template['ENBD - Classic Riders'][1:12].sum()
        report_template.loc[report_template['Description'] == 'Closing Balance at the day end', 'ENBD - Classic Riders'] = report_template['ENBD - Classic Riders'][1:12].sum() 
        cursor = conn.cursor()
        sql = f"INSERT INTO {CLOSINGBALANCETABLENAME} (ClosingBalanceDomainCompany, CreatedDate, ModifiedDate, ClosingBalance) " \
              f"VALUES (?, ?, ?, ?)"

# Prepare the values for the parameters
        params = ('ENBD-Classic-Riders', datetime.now(), datetime.now(), closingBalance)

# Execute the SQL statement with the parameters
        cursor.execute(sql, params)
        conn.commit()
        conn.close
        return report_template

def update_sql_table_for_classified(pdf_based_file_preprocessed_data, server, database, username, password, table_name):
    # Load the dataframe from Azure SQL or any other data source
    # For the purpose of this example, let's assume the dataframe is already loaded
    # into a variable named "pdf_based_file_preprocessed_data"

    # Check if the 'Classified' column is 'No'
    condition = (pdf_based_file_preprocessed_data['Classified'] == 'No')

    # Filter the dataframe to get only the rows where the condition is True
    filtered_df = pdf_based_file_preprocessed_data.loc[condition]

    # Get the IDs of the rows where the condition is True
    ids_to_update = filtered_df['TransactionId'].tolist()

    # Create the connection string
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"

    # Connect to Azure SQL
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    # Update the SQL table for each ID
    for id_to_update in ids_to_update:
        update_query = f"UPDATE {table_name} SET Classified = 'Yes' WHERE TransactionId = {id_to_update}"
        cursor.execute(update_query)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    # Return the list of IDs that were updated
    return ids_to_update

def save_dataframe_to_blob(dataframe, connection_string, container_name, excel_file_name):
    # Save DataFrame to Excel file
    report_file = BytesIO()
    dataframe.to_excel(report_file, index=False)
    report_file.seek(0)
    
    
    # Upload Excel file to Blob storage
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(excel_file_name)

    
    blob_client.upload_blob(report_file, overwrite=True, content_settings=ContentSettings(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'))

def load_nlp_models(connection_string, data_frame):
    # Remove rows with null values in the 'Narration' column
    nlp_classified = data_frame.dropna(subset=['Narration'])
    nlp_classified_data_without_nulls = nlp_classified.dropna(subset=['Narration'])
    nlp_bank_transactions = nlp_classified_data_without_nulls['Narration']

    # Load the NLP models from Azure Storage
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client("akcsaiamodel")

    # Download the model blob
    model_blob_client = container_client.get_blob_client("AkcsNlpCustommodel_V1.pkl")
    model_blob_data = model_blob_client.download_blob().readall()
    model_weights = pickle.loads(model_blob_data)

    # Download the vectorizer blob
    vectorizer_blob_client = container_client.get_blob_client("Vectorizer_V1.pkl")
    vectorizer_blob_data = vectorizer_blob_client.download_blob().readall()
    vectorizer_weights = pickle.loads(vectorizer_blob_data)

    return nlp_bank_transactions, model_weights, vectorizer_weights

def insert_data_into_training_table(server, database, username, password, pdf_based_file_preprocessed_data):
    
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
    unclassified_df = pdf_based_file_preprocessed_data[['Narration','Prediction']]
    conn = pyodbc.connect(conn_str)
    unclassifiedDatatable = "TrainingDataBankStatements"
    cursor = conn.cursor()
    sql = f"INSERT INTO {unclassifiedDatatable} (Narration, Label, CreatedDate) " \
              f"VALUES (?, ?, ?)"

# Prepare the values for the parameters
    params = []
    for index, row in unclassified_df.iterrows():
        narration = row['Narration']
        prediction = row['Prediction']
        created_date = datetime.now()
        params.append((narration, prediction, created_date))

# Execute the SQL statement with the parameters
    cursor.executemany(sql, params)
    conn.commit()
    conn.close

def main(myblob: func.InputStream):
    
    logging.info(f"Blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes")  

    
    df_template = pd.read_excel(TEMPLATEFILEPATH)
    #df_reference = pd.read_excel(template_file_path, sheet_name="term_references")
    logging.info(df_template.columns)
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTIONSTRING)
    container_client = blob_service_client.get_container_client(CONTAINERNAME)
    blobs = container_client.list_blobs()
    latest_blob = max(blobs, key=lambda blob: blob.last_modified)
    latest_blob_name = latest_blob.name
    input_file = latest_blob_name
    try:
        
        df_input_bank_statement_from_sql = fetch_data_from_sql(SERVER, DATABASE, USERNAME, PASSWORD, TRANSACTIONDETAILSTABLENAME)
        logging.info(f"Data has been fetched from {DATABASE} and the table {TRANSACTIONDETAILSTABLENAME}")
        logging.info(f"{len(df_input_bank_statement_from_sql)}")
        
        df_input_bank_statement = read_user_input_data(input_file,df_input_bank_statement_from_sql)
        logging.info(f"{len(df_input_bank_statement)} Input bank statement")
        
        preprocessed_data = general_preprocess(df_input_bank_statement)
        logging.info(f"{len(preprocessed_data)} Preprocessed data")
        
        pdf_based_file_preprocessed_data = preprocess_text_data(preprocessed_data)
        logging.info(f"{len(pdf_based_file_preprocessed_data)} PDF File preprocessed data")
        
        report_template = preprocess_template_data(df_template)
        logging.info(f"report template")
        nlp_bank_transactions, model_weights, vectorizer_weights = load_nlp_models(CONNECTIONSTRING, pdf_based_file_preprocessed_data)
        
        logging.info("Predictiing using Model path")
        predictions = predict_transactions(nlp_bank_transactions, model_weights, vectorizer_weights)

        pdf_based_file_preprocessed_data['Prediction'] = predictions
        
        #fetching unclassified data and saving it in DB
        insert_data_into_training_table(SERVER, DATABASE, USERNAME, PASSWORD, pdf_based_file_preprocessed_data)
        
        update_sql_table_for_classified(pdf_based_file_preprocessed_data, SERVER, DATABASE, USERNAME, PASSWORD, TRANSACTIONDETAILSTABLENAME)
        
        populate_report_template = populate_final_report(report_template, pdf_based_file_preprocessed_data, input_file, SERVER, DATABASE, USERNAME, PASSWORD)
        
        excel_file_name = f"output_report_{DATENOW.date()}_{DATENOW.minute}_{DATENOW.second}.xlsx"
        
        save_dataframe_to_blob(populate_report_template,CONNECTIONSTRING, OUTPUTREPORTCONTIAINERNAME, excel_file_name)

    except Exception as e:
        logging.error(e)
    
