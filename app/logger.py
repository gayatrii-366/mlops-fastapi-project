import os
import logging
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration from environment ──────────────────────
AWS_REGION       = os.getenv("AWS_REGION", "us-east-1")
LOG_GROUP        = os.getenv("CW_LOG_GROUP", "/mlops/iris-api")
LOG_STREAM       = os.getenv("CW_LOG_STREAM", "app-logs")
ENABLE_CLOUDWATCH = os.getenv("ENABLE_CLOUDWATCH", "false").lower() == "true"
AWS_ACCESS_KEY   = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY   = os.getenv("AWS_SECRET_ACCESS_KEY")

# ─── Formatter ───────────────────────────────────────────
_FORMATTER = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def setup_logger(name: str = "ml_api") -> logging.Logger:
    """
    Returns a logger that writes to:
      - stdout (always)
      - AWS CloudWatch Logs (when ENABLE_CLOUDWATCH=true and AWS creds are set)
    """
    logger = logging.getLogger(name)

    # Only configure once (avoid duplicate handlers on hot-reload)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # ── Console handler ────────────────────────────────────
    console = logging.StreamHandler()
    console.setFormatter(_FORMATTER)
    logger.addHandler(console)

    # ── CloudWatch handler ─────────────────────────────────
    if ENABLE_CLOUDWATCH:
        try:
            session_kwargs = {"region_name": AWS_REGION}
            if AWS_ACCESS_KEY and AWS_SECRET_KEY:
                session_kwargs["aws_access_key_id"]     = AWS_ACCESS_KEY
                session_kwargs["aws_secret_access_key"] = AWS_SECRET_KEY

            boto_session = boto3.session.Session(**session_kwargs)
            cw_client    = boto_session.client("logs")

            # Ensure log group exists
            try:
                cw_client.create_log_group(logGroupName=LOG_GROUP)
            except cw_client.exceptions.ResourceAlreadyExistsException:
                pass

            import watchtower
            cw_handler = watchtower.CloudWatchLogHandler(
                log_group_name=LOG_GROUP,
                log_stream_name=LOG_STREAM,
                boto3_client=cw_client,
                create_log_group=True,
            )
            cw_handler.setFormatter(_FORMATTER)
            logger.addHandler(cw_handler)
            logger.info(
                f"CloudWatch logging enabled | group={LOG_GROUP} | stream={LOG_STREAM}"
            )

        except NoCredentialsError:
            logger.warning("AWS credentials not found — CloudWatch logging disabled.")
        except ClientError as e:
            logger.warning(f"CloudWatch setup failed: {e} — logging to console only.")
        except Exception as e:
            logger.warning(f"Unexpected error setting up CloudWatch: {e}")
    else:
        logger.info("CloudWatch logging is DISABLED (set ENABLE_CLOUDWATCH=true to enable).")

    return logger