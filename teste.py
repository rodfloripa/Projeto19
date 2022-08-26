import pandas as pd
import pyodbc

# Import CSV
data = pd.read_csv('data_tecniseguros_base_jiri.csv',sep=';')  
df = pd.DataFrame(data)

# Connect to SQL Server
constr = ('DRIVER={SQL Server Native Client 10.0};Server=3.228.58.131;port=8095;Network Library=DBMSSOCN;Database=ANALITICABCK ;uid=sa;pwd=T4t1C2019;')
conn = pyodbc.connect(constr)
cursor = conn.cursor()

"""
# Create Table
cursor.execute('''
		CREATE TABLE products (
			product_id int primary key,
			product_name nvarchar(50),
			price int
			)
               ''')
"""

# Insert DataFrame to Table
for row in df.itertuples():
    cursor.execute('''
    INSERT INTO data_tecniseguros_base_jiri (compro,delta, 
    es_renovable,fin_vigencia,id_aseguradora,
    id_ciudad,id_cliente,id_cliente_tipo,id_producto_ciudad,
    id_ramo,id_tipo_cliente,margen_contribucion,
    nombre_aseguradora,nombre_ciudad,nombre_ramo,poliza,
    produccion,renovo,exito_tiempo,exito_llamada,grupo,
    predict,id)
    ''',
    row.compro,
    row.delta,
    row.es_renovable,
    row.fin_vigencia,
    row.id_aseguradora,
    row.id_ciudad,
    row.id_cliente,
    row.id_cliente_tipo,
    row.id_producto_ciudad,
    row.id_ramo,
    row.id_tipo_cliente,
    row.margen_contribucion,
    row.nombre_aseguradora,
    row.nombre_ciudad,
    row.nombre_ramo,
    row.poliza,
    row.produccion,
    row.renovo,
    row.exito_tiempo,
    row.exito_llamada,
    row.grupo,
    row.predict,
    row.id
    )
conn.commit()