# ISLES 2022 hidden-test submission 1: nnU-Net 5-fold ensemble (see inference.py).
# Build on a machine with Docker, from revision/container/, after running prepare_model.sh:
#   docker build -t strokebench-isles22-nnunet .
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN pip install --no-cache-dir nnunetv2==2.8.1 dynamic_network_architectures==0.4.4 \
        batchgenerators==0.25.6 batchgeneratorsv2==0.3.8 simpleitk==2.5.7 numpy==1.26.4

COPY stroke_trainer_stub.py /tmp/stroke_trainer_stub.py
RUN cp /tmp/stroke_trainer_stub.py "$(python -c 'import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))')/training/nnUNetTrainer/variants/stroke_trainer_stub.py"

WORKDIR /opt/algorithm
COPY --chown=user:user model/ /opt/algorithm/model/
COPY --chown=user:user inference.py /opt/algorithm/inference.py
ENV nnUNet_raw=/tmp nnUNet_preprocessed=/tmp nnUNet_results=/tmp nnUNet_compile=f
USER user
ENTRYPOINT ["python", "-u", "/opt/algorithm/inference.py"]
