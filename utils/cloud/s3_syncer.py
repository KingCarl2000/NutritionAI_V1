
import subprocess

class S3Sync:
    def sync_folder_to_s3(self, folder, aws_bucket_url):
        command = f"aws s3 sync {folder} {aws_bucket_url}"
        # Adding shell=True allows Windows shell to resolve 'aws' executable
        subprocess.run(command, shell=True, check=True)

    def sync_folder_from_s3(self, folder, aws_bucket_url):
        command = f"aws s3 sync {aws_bucket_url} {folder}"
        # Adding shell=True allows Windows shell to resolve 'aws' executable
        subprocess.run(command, shell=True, check=True)