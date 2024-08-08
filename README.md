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





