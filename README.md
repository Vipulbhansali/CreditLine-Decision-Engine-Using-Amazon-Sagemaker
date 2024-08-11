# XGBoost Machine Learning Project using Amazon SageMaker #


This project demonstrates the implementation of a complete machine learning pipeline using XGBoost on Amazon SageMaker.

## Table of Contents

The project has been developed over following steps :

- Step 1: Project & Variable Setup
- Step 2: Splitting and Transforming the Data
- Step 3: Training the Model
- Step 4: Tuning the Model
- Step 5: Evaluating the Model
- Step 6: Registering the Model
- Step 7: Conditional Registration
- Step 8: Serving the Model
- Step 9: Deploying the Model
- Step 10: Deploying From the Pipeline
- Step 11: Deploying From an Event
- Step 12: Building an Inference Pipeline
- Step 13: Data Quality Baseline
- Step 14: Model Quality Baseline
- Step 15: Data Monitoring
- Step 16: Model Monitoring
- Step 17: Shadow Deployments

## Project Overview

**Objective**: The objective of this project is to build, train, evaluate, and deploy an XGBoost model using Amazon SageMaker to predict liklihood of extending Personal loan creditline based on various personal and financial factors.

## Architecture and Setup

**Environment Setup**: Ensure the following are set up:

- Instance above ml.m5.xarge
- SageMaker execution role
- S3 bucket for data storage
- Necessary IAM policies
- XGBoost image
- Boto3 and IAM client setup

## Experiment Tracking with Comet

This project uses Comet for tracking experiments, visualizing metrics, and logging artifacts. Comet provides an easy-to-use interface to monitor the progress of machine learning experiments.


## Split and Transform

This step uses an AWS SageMaker 'Processing Job' with an SKLearn processor to split the dataset into training, validation, and test sets, apply necessary transformations, save the data splits to S3, and store the transformation pipelines in a `model.tar.gz` file.


## Training

This step uses an AWS SageMaker estimator to train the machine learning model. The estimator handles the training process, and the trained model is saved to S3 for future use.


## Tuning

This step uses an AWS SageMaker tuner to optimize the hyperparameters of the machine learning model. The tuner explores various hyperparameter configurations by running a hyperparameter tuning job to find the best set of parameters, and the results are saved to S3.


## Evaluation

This step uses an AWS SageMaker processing job to evaluate the machine learning model. It performs the following tasks:
 - Loads the test data and the trained model into memory, and evaluates the model to generate metrics like accuracy and precision, saving the results in `evaluation.json` and uploading it to S3.
 - Maps the `evaluation.json` output to a property file for easy retrieval in the pipeline.
 - Uses a custom Docker image that includes both the SKLearn processor and XGBoost, uploaded to a registry and used for the processing container.


## Model Registration

This step involves registering the trained model in the AWS SageMaker Model Registry. It includes the following tasks:

-  Creates an instance of the built-in XGBoost class to be used as the container for the model.
-  Populates the model's metrics, generated during the evaluation step, using the `ModelMetrics` class.
-  Uses the `ModelStep` to register the model in the Model Registry, making it available for deployment and further use.

## Conditional Registration

This step adds conditional logic to the model registration process in the AWS SageMaker pipeline. It includes the following tasks:

- Sets a performance threshold for the model and defines a fail step using the `FailStep` class from SageMaker.
- Creates a condition using the `sagemaker.workflow.condition.Condition` class to check if the model meets the threshold.
- Uses a `ConditionStep` to determine whether to proceed with model registration or trigger the fail step, and adds this logic to the pipeline.


## Serving the Model

This step involves deploying the top-performing model locally. It includes the following tasks:
- Retrieves the top-performing model from the SageMaker Model Registry using the SageMaker client (`boto3` API) and downloads it.
- Builds a simple Flask application as a wrapper around the model.
- Serves the model locally using the Flask app, enabling it to handle inference requests.


## Deploy the Model

This step involves deploying the registered model in AWS SageMaker. It includes the following tasks:
- Creates a model package using the `ModelPackage` class from SageMaker.
- Utilizes the ARN of the model registered in the Model Registry to specify which model to deploy.
- Deploys the model, making it available for inference in the specified environment.
- Then we test the Endpoint by sending a payload, using Predictor class by sagemaker.


## Deploying from the Pipeline

This step automates the deployment of the model from the pipeline with data capture enabled. 
It includes the following tasks:
- Enables Data Capture as part of the SageMaker endpoint configuration, capturing all input data and the corresponding predictions.
- Sets up a Lambda function to handle the deployment of the model, triggered as part of the pipeline.
- The Lambda function is invoked by the pipeline to deploy the model to the specified endpoint, ensuring seamless integration with the overall workflow.

## Deploying from the Event (Human in the Loop Deployment)

This step integrates human-in-the-loop deployment by triggering model deployment through an event. It includes the following tasks:
- Configures Amazon EventBridge to trigger a Lambda function by creating an EventBridge rule.
- Sets the Lambda function as the target for the EventBridge rule and updates its permissions to allow triggering by the rule.
- Updates the pipeline steps to incorporate this event-driven deployment process, allowing for human intervention before the model is deployed.











