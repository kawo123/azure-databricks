# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Spark integration with Azure Cognitive Services
# MAGIC 
# MAGIC At Spark + AI Summit 2019, Microsoft announced a new set of models in the SparkML ecosystem that make it easy to leverage the Azure Cognitive Services at terabyte scales. With only a few lines of code, developers can embed cognitive services within your existing distributed machine learning pipelines in Spark ML. Additionally, these contributions allow Spark users to chain or Pipeline services together with deep networks, gradient boosted trees, and any SparkML model and apply these hybrid models in elastic and serverless distributed systems.
# MAGIC 
# MAGIC From image recognition to object detection using speech recognition, translation, and text-to-speech, Azure Cognitive Services makes it easy for developers to add intelligent capabilities to their applications in any scenario. This notebook demostrates the integration of PySpark (using Azure Databricks) with Azure Cognitive Service [Text Analytics](https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/) to extract valuable information from text data. 
# MAGIC 
# MAGIC ## Prerequisites
# MAGIC 
# MAGIC - Spark 2.4 environment
# MAGIC   - You can use Azure Databricks for an integrated Spark environment
# MAGIC - Install required libraries in Spark
# MAGIC   - [MMLSpark](https://mmlspark.blob.core.windows.net/website/index.html#install)
# MAGIC   - [azure-cognitiveservices-language-textanalytics](https://pypi.org/project/azure-cognitiveservices-language-textanalytics/)
# MAGIC - Create [Azure Cognitive Services multi-service resource](https://docs.microsoft.com/en-us/azure/cognitive-services/cognitive-services-apis-create-account?tabs=multiservice%2Clinux)
# MAGIC - Import [Customers sample dataset](https://github.com/kawo123/azure-databricks/blob/master/data/customers.csv) into Spark environment
# MAGIC 
# MAGIC ## References
# MAGIC 
# MAGIC - [Spark and Azure Cognitive Services blog](https://azure.microsoft.com/en-us/blog/dear-spark-developers-welcome-to-azure-cognitive-services/)

# COMMAND ----------

from azure.cognitiveservices.language.textanalytics import TextAnalyticsClient
from msrest.authentication import CognitiveServicesCredentials
from mmlspark.cognitive import TextSentiment
from pyspark.sql.functions import col

# COMMAND ----------

# Obtain Azure Text Analytics endpoint and key. Replace <<TODO>> below with your endpoint and key
textanalytics_endpoint = '<<TODO>>' # TODO
textanalytics_key = '<<TODO>>' # TODO

# Initialize Azure Text Analytics client
client = TextAnalyticsClient(textanalytics_endpoint, CognitiveServicesCredentials(textanalytics_key))

# COMMAND ----------

# Create sample text documents for analysis
docs = [
  { 'id': '1', 'language': 'en', 'text': 'This is awesome!' },
  { 'id': '2', 'language': 'en', 'text': 'This was a waste of my time. The speaker put me to sleep.' },
  { 'id': '3', 'language': 'en', 'text': None },
  { 'id': '4', 'language': 'en', 'text': 'Hello World' }
]

# Submit text documents for sentiment analysis
resp = client.sentiment(documents=docs)

# Print sentiment analysis results
for document in resp.documents:
    print("Document Id: ", document.id, ", Sentiment Score: ", "{:.2f}".format(document.score))

# COMMAND ----------

# MAGIC %md
# MAGIC You should observe output similar to below
# MAGIC 
# MAGIC ```
# MAGIC Document Id:  1 , Sentiment Score:  1.00
# MAGIC Document Id:  2 , Sentiment Score:  0.11
# MAGIC Document Id:  4 , Sentiment Score:  0.76
# MAGIC ```

# COMMAND ----------

# Read customers csv
df_customers = spark\
  .read\
  .option('header', True)\
  .option('inferSchema', True)\
  .csv('/FileStore/tables/customers.csv')

df_customers.show(2)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC You should see a table with the following columns: `Customer grouping`, `Product ID`, `State`, `Customer category`, `Product market price`, `total market price`, `Notes`, `store comments`, `Customer Address`, `Gender`, `Discount`, `Date`, `Quantity`, `Discount_discrete`

# COMMAND ----------

# Define Sentiment Analysis pipeline
pipe_text_sentiment = (TextSentiment()
  .setSubscriptionKey(textanalytics_key)
  .setLocation('eastus')
  .setLanguage('en')
  .setTextCol('store comments')
  .setOutputCol("StoreCommentSentimentObj")
  .setErrorCol("Errors")
  .setConcurrency(10)
)

# Process df_customers with the Sentiment Analysis pipeline 
df_customers_sentiment = pipe_text_sentiment.transform(df_customers)

df_customers_sentiment.show(2)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC After the customer dataframe is processed by the sentiment analysis pipeline, you should see 2 additional columns in the table: `Errors` and `StoreCommentSentimentObj`. The `Errors` column contains any error message that the text analytics pipeline encounter. The `StoreCommentSentimentObj` column contains the an array of sentiment objects returned by the Text Analytics service. The sentiment object includes sentiment score and any error messages that the Text Analytics engine encounters.

# COMMAND ----------

# Extract sentiment score from store comment sentiment complex objects
df_customers_sentiment_numeric = (df_customers_sentiment
  .select('*', col('StoreCommentSentimentObj').getItem(0).getItem('score').alias('StoreCommentSentimentScore'))
  .drop('Errors', 'StoreCommentSentimentObj')
)

df_customers_sentiment_numeric.show(2)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC You should now see a new column `StoreCommentSentimentScore` which contains the numeric sentiment scores of store comments
