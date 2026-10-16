import os

# Must be set before Prefect is imported anywhere.
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")   # no telemetry
os.environ.setdefault("PREFECT_LOGGING_LEVEL", "WARNING")            # quiet test output
