FROM pytorch/pytorch

WORKDIR /working_dir/Floor_plan_generation_using_GNNs

COPY . ./

SHELL ["/bin/bash", "-c"]

RUN pip install -r requirments.txt

ENTRYPOINT [ "/bin/bash", "-c" ]
CMD ["Docker/Run.sh"]

LABEL maintainer='Mohamed Hassan'
LABEL description='Graduation project, Residential Floor Plan Generation \
Using Deep Learning Techniques [specially room size estimation with Graph Neural Networks].'

