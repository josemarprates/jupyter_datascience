
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, count, mean, stddev, expr
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, StringIndexer
from pyspark.ml.linalg import Vectors

# Criar uma sessão Spark
spark = SparkSession.builder.appName("PrevisaoColheitaAgricola").getOrCreate()


# Carregar o dataset
dataset_path = "./opt/spark/dados/dataset_agricola.csv"  # Atualize para o caminho correto do arquivo

df = spark.read.csv(dataset_path, header=True, inferSchema=True)

# Descrição do dataset
df.printSchema()

# Verificar valores ausentes
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# Normalizar nomes das colunas
df = df.toDF(*[c.lower().replace(" ", "_").replace("-", "_") for c in df.columns])

# Identificar outliers com IQR
numeric_columns = [c for c, t in df.dtypes if t in ("int", "double")]
bounds = {}

for col_name in numeric_columns:
    quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.05)
    Q1, Q3 = quantiles
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    bounds[col_name] = (lower_bound, upper_bound)

# Contar o número de outliers
outliers_count = {col: df.filter((col(col_name) < bounds[col][0]) | (col(col_name) > bounds[col][1])).count()
                  for col in numeric_columns}
print("Outliers count:", outliers_count)

# Aplicar normalização (Min-Max Scaling)
assembler = VectorAssembler(inputCols=numeric_columns, outputCol="features")
scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
pipeline = Pipeline(stages=[assembler, scaler])
df_scaled = pipeline.fit(df).transform(df)

# Converter coluna de tipo de cultura para index numérico
indexer = StringIndexer(inputCol="tipo_de_cultura", outputCol="cultura")
df = indexer.fit(df).transform(df)

# Preencher valores ausentes em cultura com -1
df = df.withColumn("cultura", when(df["cultura"].isNull(), -1).otherwise(df["cultura"]))

# Mostrar os dados processados
df.show()
