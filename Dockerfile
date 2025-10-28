ARG CANN_VERSION=8.2.rc1
ARG DEVICE_TYPE=910b
ARG OS=openeuler22.03
ARG PYTHON_VERSION=py3.11

FROM quay.io/ascend/cann:$CANN_VERSION-$DEVICE_TYPE-$OS-$PYTHON_VERSION

# Update pip & apt sources
ARG DEVICE_TYPE
#ARG PIP_INDEX_URL="https://pypi.org/simple/"
ARG APTMIRROR=""
ARG MEMFABRIC_URL=https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/mf_adapter-1.0.0-cp311-cp311-linux_aarch64.whl
ARG PYTORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ARG PTA_URL="https://gitcode.com/Ascend/pytorch/releases/download/v7.1.0.2-pytorch2.6.0/torch_npu-2.6.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl"
ARG VLLM_TAG=v0.8.5
ARG TRITON_ASCEND_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/triton_ascend-3.2.0%2Bgitb0ea0850-cp311-cp311-linux_aarch64.whl"
ARG BISHENG_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/Ascend-BiSheng-toolkit_aarch64.run"
ARG SGLANG_TAG=v0.5.4
ARG ASCEND_CANN_PATH=/usr/local/Ascend/ascend-toolkit
ARG SGLANG_KERNEL_NPU_TAG=main

USER root

# Define environments
ENV DEBIAN_FRONTEND=noninteractive

#RUN pip config set global.index-url $PIP_INDEX_URL
RUN if [ -n "$APTMIRROR" ];then sed -i "s|.*.ubuntu.com|$APTMIRROR|g" /etc/apt/sources.list ;fi

# ʹ��DNF���APT���а�������������RPM����
RUN dnf update -y && \
    dnf install -y \
        gcc gcc-c++ make autoconf automake libtool \
        cmake \
        vim \
        wget \
        curl \
        net-tools \
        zlib-devel \
        lld \
        clang \
        glibc-locale-source \
        ccache \
        openssl \
		openssl-devel \
		pkgconf \
		ca-certificates \
		protobuf-devel protobuf-compiler && \
	rm -f /usr/lib/locale/locale-archive && \
	localedef -c -f UTF-8 -i en_US en_US.UTF-8 && \
    dnf clean all && \  
    rm -rf /var/cache/dnf/* 
    

# ������������
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
ENV PATH="/root/.cargo/bin:${PATH}"


# Install dependencies
# TODO: install from pypi released memfabric
RUN pip install $MEMFABRIC_URL --no-cache-dir

RUN pip install setuptools-rust wheel build --no-cache-dir

# install rustup from rustup.rs
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rustc --version && cargo --version && protoc --version

# Install vLLM
RUN git clone --depth 1 https://github.com/vllm-project/vllm.git --branch $VLLM_TAG && \
    (cd vllm && VLLM_TARGET_DEVICE="empty" pip install -v . --no-cache-dir) && rm -rf vllm

# TODO: install from pypi released triton-ascend
RUN pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/cpu --no-cache-dir \
    && wget ${PTA_URL} && pip install "./torch_npu-2.6.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl" --no-cache-dir \
    && python3 -m pip install --no-cache-dir attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml pybind11 \
    && pip install ${TRITON_ASCEND_URL} --no-cache-dir

# Install SGLang
RUN git clone https://github.com/sgl-project/sglang --branch $SGLANG_TAG && \
    (cd sglang/python && rm -rf pyproject.toml && mv pyproject_other.toml pyproject.toml && pip install -v .[srt_npu] --no-cache-dir) && \
    pip install sglang-router && \
    rm -rf sglang

	
# Install Deep-ep
# pin wheel to 0.45.1 ref: https://github.com/pypa/wheel/issues/662
RUN pip install wheel==0.45.1 && git clone  --branch $SGLANG_KERNEL_NPU_TAG  https://github.com/sgl-project/sgl-kernel-npu.git \
    && export LD_LIBRARY_PATH=${ASCEND_CANN_PATH}/latest/runtime/lib64/stub:$LD_LIBRARY_PATH \
    && source ${ASCEND_CANN_PATH}/set_env.sh  \
    && cd sgl-kernel-npu  \
    && bash build.sh \
    && pip install output/deep_ep*.whl output/sgl_kernel_npu*.whl --no-cache-dir \
    && cd .. && rm -rf sgl-kernel-npu \
    && cd "$(pip show deep-ep | awk '/^Location:/ {print $2}')" && ln -s deep_ep/deep_ep_cpp*.so

# Install CustomOps
RUN wget https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ops/CANN-custom_ops-8.2.0.0-$DEVICE_TYPE-linux.aarch64.run && \
    chmod a+x ./CANN-custom_ops-8.2.0.0-$DEVICE_TYPE-linux.aarch64.run && \
    ./CANN-custom_ops-8.2.0.0-$DEVICE_TYPE-linux.aarch64.run --quiet --install-path=/usr/local/Ascend/ascend-toolkit/latest/opp && \
    wget https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ops/custom_ops-1.0.$DEVICE_TYPE-cp311-cp311-linux_aarch64.whl && \
    pip install ./custom_ops-1.0.$DEVICE_TYPE-cp311-cp311-linux_aarch64.whl

# Install Bisheng
RUN wget ${BISHENG_URL} && chmod a+x Ascend-BiSheng-toolkit_aarch64.run && ./Ascend-BiSheng-toolkit_aarch64.run --install && rm Ascend-BiSheng-toolkit_aarch64.run

# #�������
# ENV http_proxy=
# ENV https_proxy=
# ENV all_proxy=

# # ����LABEL
# LABEL cmb.cs.baseimage.name="cmb-nexinfer" \
#       cmb.cs.baseimage.version="v0.5.4" \
#       cmb.cs.baseimage.tag="aarch64-hw-20251024"
	

# # �����û� aiuser��UID=1001��GID=0��root �飩
# RUN useradd -u 1001 -g 0 -m aiuser && \
#     mkdir -p /home/aiuser/.local && \
#     chown -R 1001:0 /home/aiuser


# # ���ƻ������������ļ�
# COPY start_env.sh /home/aiuser/
# RUN chmod +x /home/aiuser/start_env.sh

# # ����modelet����Ȩ
# COPY modelet /usr/local/bin/
# RUN chmod +x /usr/local/bin/modelet

# # ����û�
# USER aiuser

# # ָ������Ŀ¼
# WORKDIR /home/aiuser

# CMD ["/bin/bash","start_env.sh"]
