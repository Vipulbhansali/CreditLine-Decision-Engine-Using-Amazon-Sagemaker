FROM continuumio/miniconda3

# Install scikit-learn and XGBoost
RUN conda install -y scikit-learn xgboost

# Install additional dependencies if needed
RUN pip install sagemaker

# Copy your evaluation script into the container
COPY script.py /opt/program/script.py

# Set the working directory
WORKDIR /opt/program

# Specify the entry point for the container
ENTRYPOINT ["python3", "script.py"]
