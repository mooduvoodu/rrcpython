import pyodbc  # needed to connect to a read database server over ODBC
import keyring  # used to hide secrets from being clear text in code
import pandas as pd

#keyring.set_password("sqlserverpass", "srvadmin", "Palmetto7")  #remove once set

server = 'pdbserver.database.windows.net'
database = 'sampledb'
username = 'srvadmin'
password = keyring.get_password("sqlserverpass", "srvadmin")
driver= '{ODBC Driver 17 for SQL Server}'



with pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password) as conn:
    SQL_Query = pd.read_sql_query(
    '''SELECT ProductID,Name 
    FROM [SalesLT].[Product]''', conn)
    var1 = 'test'
    df = pd.DataFrame(SQL_Query, columns=['ProductID','Name'])
    print (df)


