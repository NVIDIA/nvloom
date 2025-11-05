# This Dockerfile is only a sample. You likely want to customize it for your needs.
# You don't need to use containers to run nvloom.

FROM nvidia/cuda:12.9.0-devel-ubuntu24.04 AS base

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && apt-get install -y --no-install-recommends cmake libopenmpi-dev libopenmpi3t64 libboost-program-options-dev

ADD . /usr/local/src/nvloom

RUN cd /usr/local/src/nvloom && \
    cmake . && make -j16 VERBOSE=1 && \
    install -m 755 nvloom_cli /usr/local/bin

RUN install -m 755 /usr/local/src/nvloom/cli/plot_heatmaps.py /usr/local/bin

FROM nvidia/cuda:12.9.0-base-ubuntu24.04

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && apt-get install -y --no-install-recommends openmpi-bin libopenmpi3t64 python3 python3-matplotlib

COPY --from=base /usr/local/bin/nvloom_cli /usr/local/bin
COPY --from=base /usr/local/bin/plot_heatmaps.py /usr/local/bin
