ARG TAG=v4.2.5
FROM ghcr.io/flatland-association/flatland-baselines:${TAG}

COPY submission/ submission/

ENV POLICY=submission.my_policy.MyPolicy
ENV OBS_BUILDER=submission.my_observation_builder.MyObservationBuilder

RUN bash entrypoint_generic.sh python -m pip install -r submission/requirements.txt