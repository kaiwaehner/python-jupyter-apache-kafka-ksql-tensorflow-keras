# Work in progress - NOT FINISHED YET

## Use Case: Fraud Detection for Credit Card Payments
We use test data set from Kaggle as foundation to train an unsupervised autoencoder to detect anomalies and potential fraud in payments. 

Focus of this project is not just model training, but the whole Machine Learning infrastructure including data ingestion, data preprocessing, model training, model deployment and monitoring. All of this needs to be scalable, reliable and performant.

## Technology: Python, Jupyter, TensorFlow, Keras, Apache Kafka, KSQL 
This project shows a demo which combines

- simplicity of data science tools (Python, Jupyter notebooks, NumPy, Pandas)
- powerful Machine Learning / Deep Learning frameworks (TensorFlow, Keras)
- reliable, scalable event-based streaming technology for production deployments (Apache Kafka, Kafka Connect, KSQL).

## Requirements

- Python (tested with 3.6)
- Java 8+ (tested with Java 8)
- [Confluent Platform 5.0+ using Kafka + KSQL](https://www.confluent.io/download/) (tested with 5.1)
- [ksql-python](https://github.com/bryanyang0528/ksql-python) (tested with Github release 5.x released on 2018-10-12)

### ksql-python Installation

If you have problems installing ksql-python in your environment via 'pip install ksql', use the commands described in the Github project instead. 

After installation, for some reason, the 'from ksql import KSQLAPI' statement did not work with Python 2.7.x in my Jupyter notebook (but in Mac terminal), so I used Python 3.6 (which also worked in Jupyter).

## Step-by-step guide

We will do the following:

1) *Data Integration (Kafka Connect)*: Integrate a stream of data from CSV file or continuous data stream (in real world you can connect directly to an existing Kafka stream from the Jupyter notebook). As alternative, you can create new events manually in command line
2) *Data Preprocessing (KSQL)*: Preprocess the data, e.g. filter, anonymize, aggreate / concatenate
3) *Machine Learning specific preprocessing (NumPy, Pandas, Scikit-learn): Normalize, split train / test data
4) *Model Training (TensorFlow + Keras)*
5) *Model Deployment (KSQL + Tensorflow)*
6) *Monitoring of Model Behaviour (KSQL)* like accuracy and performance 

While all of this can be done in a Jupyter notebook for interactive analysis, we can then deploy the same pipeline to production at scale. For instance, you can re-use the KSQL preprocessing statements and run them in your production infrastructure to do model inference with KSQL and the TensorFlow model at scale on new incoming event streams.

Check out [this step-by-step guide](https://github.com/kaiwaehner/python-jupyter-apache-kafka-ksql-tensorflow-keras/blob/master/live-demo___python-jupyter-apache-kafka-ksql-tensorflow-keras.adoc) to start the backend and notebook. The main demo is running in the Jupyter notebook 'python-jupyter-apache-kafka-ksql-tensorflow-keras.ipynb' afterwards.


## Autoencoder for Credit Card Fraud Detection build with Keras and TensorFlow

An autoencoder is an unsupervised neural network which encodes (i.e. compresses) the input and then decodes (i.e. decompresses) it again:

![Autoencoder (Unsupervised neural network)](pictures/AutoEncoder.png)

The goal is to lose as little information as possible. This way we can use an autoencoder to detect anomalies if the decoding cannot reconstruct the input well (showing potential fraud).  

## Hands-On with Python, TensorFlow, Keras, Apache Kafka and KSQL

We use KSQL for preprocessing, Numpy, Pandas and scikit-learn for ML-specific tasks like array shapes or splitting training and test data, TensorFlow + Keras for model training, and Kafka Streams or KSQL for model deployment and monitoring.

Here is a TensorBoard screenshot of the trained Autoencoder:

![Autoencoder for Fraud Detection (TensorBoard)](pictures/Keras_TesnsorFlow_Autoencoder_Fraud_Detection_TensorBoard.png)

## Apache Kafka and KSQL within a Jupyter notebook

Interactive analysis and data-preprocessing with Python and KSQL:

![KSQL + Python for Interactive Data Processing](pictures/Apache_Kafka_KSQL_Python_Jupyter_Notebook.png)

## Keras model (.h5) vs. TensorFlow model (.pb)

If you want to deploy the model in a TensorFlow infrastructure like Google ML Engine, it is best to train the model with GCP's tools as describe in this [Google ML Getting Started] (https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction) guide.

Otherwise you need to convert the H5 Keras file to a TensorFlow Protobuffers file and fulfil some more tasks, e.g. described in this [blog post](https://medium.com/google-cloud/serve-keras-models-using-google-cloud-machine-learning-services-910912238bf6).

The Python tool [Keras to TensorFlow](https://github.com/amir-abdi/keras_to_tensorflow) is a good and simple solution:

                python keras_to_tensorflow.py --input_model="/Users/kai.waehner/git-projects/python-jupyter-apache-kafka-ksql-tensorflow-keras/models/autoencoder_fraud.h5" --output_model="/Users/kai.waehner/git-projects/python-jupyter-apache-kafka-ksql-tensorflow-keras/models/autoencoder_fraud.pb"

The tool freezes the nodes (converts all TF variables to TF constants), and saves the inference graph and weights into a binary protobuf (.pb) file.

TODO Use keras.estimator.model_to_estimator (included in tf.keras)? Example: https://www.kaggle.com/yufengg/emnist-gpu-keras-to-tf