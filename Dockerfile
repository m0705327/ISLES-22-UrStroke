FROM python:3.6.13


RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip
RUN pip install opencv-python-headless
RUN pip install h5py==2.10.0



COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -rrequirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm best1.hdf5 /opt/algorithm/
COPY --chown=algorithm:algorithm best2.hdf5 /opt/algorithm/
COPY --chown=algorithm:algorithm best3.hdf5 /opt/algorithm/


ENTRYPOINT python -m process $0 $@
