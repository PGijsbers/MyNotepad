# A dockerfile which generates the environment in which runtime_prediction.py works.
# Taken https://github.com/openml/automlbenchmark/ from and modified.
FROM ubuntu:16.04

# == Python setup ==
RUN apt-get update
RUN apt-get update
RUN apt-get install -y python3.5 python3-pip
RUN pip3 install --upgrade pip

# NumPy and SciPy need be installed first, otherwise OpenML setup will fail.
RUN pip3 install --no-cache-dir numpy>=1.15.0
RUN pip3 install --no-cache-dir scipy>=1.1.0
RUN pip3 install --no-cache-dir openml>=0.7.0
RUN pip3 install --no-cache scikit-learn>=0.19.1
RUN pip3 install --no-cache pandas>=0.23.4
RUN pip3 install --no-cache cvxpy>=1.0.6

RUN apt-get install -y git


WORKDIR /runtime/
RUN git clone https://github.com/udellgroup/oboe.git
ADD runtime_prediction.py  /code/runtime_prediction.py

#ENV start_call="python3 run_oboe.py"

# https://docs.docker.com/engine/reference/builder/#entrypoint
#ENTRYPOINT ["/bin/bash", "-c", "./common/startup.sh -c \"$start_call\" $0 $*"]
