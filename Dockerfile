ARG TAG=v4.2.5
FROM ghcr.io/flatland-association/flatland-baselines:${TAG}

# کپی کردن پوشه کدها به داکر (دقیقاً در مسیر پیش‌فرضِ خودِ مسابقه)
COPY submission/ submission/

# تنظیمات محیطی مسابقه
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV POLICY=submission.my_policy.MyPolicy
ENV OBS_BUILDER=submission.my_observation_builder.MyObservationBuilder

# نصب پکیج‌ها با اسکریپت پیش‌فرض خودِ ایمیج تا محیط کاندا کاملاً فعال بماند
RUN bash entrypoint_generic.sh python -m pip install -r submission/requirements.txt