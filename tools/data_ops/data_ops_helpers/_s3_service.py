import boto3
from tools.constant import S3_REGION

class S3Service():
    def get_s3_client(self) -> boto3.session.Session.client:
        global _s3_client
        if not _s3_client:
            _s3_client = boto3.client("s3", region_name=S3_REGION)
        return _s3_client

    def upload_file_to_s3(self, filepath: str, bucket: str, key: str, ExtraArgs={}):
        s3_client = self.get_s3_client()
        s3_client.upload_file(
            Filename=filepath, Bucket=bucket, Key=key, ExtraArgs={**ExtraArgs}
        )