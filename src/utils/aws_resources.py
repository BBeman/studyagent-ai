"""
AWS Resource Management for StudyAgent AI
Handles S3 document storage operations.


Architecture:
- S3 bucket stores uploaded documents (PDFs, notes, etc.)
- AgentCore Memory handles conversation context (managed separately)
- Documents are read from S3 and passed as context to the agent
"""
import os
import io
import logging

from botocore.exceptions import ClientError

from src.config import AWS_REGION, S3_BUCKET_NAME, ANALYTICS_TABLE_NAME, get_boto3_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aws-resources")


class AWSResourceManager:
    """
    Manages AWS resources for StudyAgent AI.
    Handles S3 document storage operations.
    """

    def __init__(self):
        """Initialize the AWS resource manager."""
        self.session = get_boto3_session()
        self.s3 = self.session.client("s3")

    # =========================================================================
    # S3 Bucket Management
    # =========================================================================
    def create_s3_bucket(self, bucket_name: str = S3_BUCKET_NAME) -> str:
        """Create S3 bucket for document storage if it doesn't exist."""
        try:
            self.s3.head_bucket(Bucket=bucket_name)
            logger.info(f"S3 bucket already exists: {bucket_name}")
            return bucket_name
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.info(f"Creating S3 bucket: {bucket_name}")

                if AWS_REGION == "us-east-1":
                    self.s3.create_bucket(Bucket=bucket_name)
                else:
                    self.s3.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": AWS_REGION},
                    )

                self.s3.put_bucket_versioning(
                    Bucket=bucket_name,
                    VersioningConfiguration={"Status": "Enabled"},
                )

                for folder in ["documents/", "modules/"]:
                    self.s3.put_object(Bucket=bucket_name, Key=folder)

                logger.info(f"S3 bucket created: {bucket_name}")
                return bucket_name
            else:
                raise

    def upload_document_to_s3(
        self, file_path: str, module: str, bucket_name: str = S3_BUCKET_NAME
    ) -> str:
        """Upload a document to S3, organized by module."""
        filename = os.path.basename(file_path)
        s3_key = f"documents/{module.replace(' ', '_').lower()}/{filename}"

        logger.info(f"Uploading {filename} to s3://{bucket_name}/{s3_key}")
        self.s3.upload_file(file_path, bucket_name, s3_key)

        return f"s3://{bucket_name}/{s3_key}"

    def get_document_content(
        self, s3_key: str, bucket_name: str = S3_BUCKET_NAME
    ) -> str:
        """Get document content from S3. Handles PDFs, DOCX, and text files."""
        try:
            response = self.s3.get_object(Bucket=bucket_name, Key=s3_key)
            content = response["Body"].read()
            ext = s3_key.lower().split(".")[-1]

            if ext == "pdf":
                try:
                    from pypdf import PdfReader

                    pdf_reader = PdfReader(io.BytesIO(content))
                    text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    return text
                except Exception as e:
                    logger.warning(f"Could not parse PDF {s3_key}: {e}")
                    return f"[PDF document: {s3_key}]"
            elif ext == "docx":
                try:
                    from docx import Document

                    doc = Document(io.BytesIO(content))
                    return "\n\n".join(
                        para.text for para in doc.paragraphs if para.text.strip()
                    )
                except Exception as e:
                    logger.warning(f"Could not parse DOCX {s3_key}: {e}")
                    return f"[DOCX document: {s3_key}]"
            else:
                try:
                    return content.decode("utf-8")
                except UnicodeDecodeError:
                    return f"[Binary document: {s3_key}]"
        except Exception as e:
            logger.error(f"Error getting document: {e}")
            return ""

    def list_documents(
        self, module: str = None, bucket_name: str = S3_BUCKET_NAME
    ) -> list:
        """List documents in S3, optionally filtered by module."""
        prefix = "documents/"
        if module:
            prefix = f"documents/{module.replace(' ', '_').lower()}/"

        response = self.s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        documents = []
        for obj in response.get("Contents", []):
            if not obj["Key"].endswith("/"):
                key_parts = obj["Key"].split("/")
                documents.append({
                    "key": obj["Key"],
                    "filename": key_parts[-1] if key_parts else obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                    "module": key_parts[1] if len(key_parts) > 1 else "unknown",
                })

        return documents

    def delete_document(self, s3_key: str, bucket_name: str = S3_BUCKET_NAME):
        """Delete a document from S3."""
        self.s3.delete_object(Bucket=bucket_name, Key=s3_key)
        logger.info(f"Deleted: s3://{bucket_name}/{s3_key}")

    def get_all_document_contents(
        self,
        module: str = None,
        bucket_name: str = S3_BUCKET_NAME,
        max_chars: int = 50000,
    ) -> str:
        """Get contents of all documents as a single context string."""
        documents = self.list_documents(module=module, bucket_name=bucket_name)

        context_parts = []
        total_chars = 0

        for doc in documents:
            if total_chars >= max_chars:
                break

            content = self.get_document_content(doc["key"], bucket_name)
            if content and not content.startswith("["):
                remaining = max_chars - total_chars
                if len(content) > remaining:
                    content = content[:remaining] + "...[truncated]"

                filename = doc.get("filename", doc["key"].split("/")[-1])
                module_name = doc.get("module", "unknown").replace("_", " ").title()
                doc_header = f"\n\n{'='*60}\n[START OF {filename} | Module: {module_name} | {len(content)} characters of content follow]\n{'='*60}\n"
                doc_footer = f"\n[END OF {filename}]\n"
                context_parts.append(doc_header + content + doc_footer)
                total_chars += len(doc_header) + len(content) + len(doc_footer)

        return "".join(context_parts) if context_parts else ""

    def create_analytics_table(self, table_name: str = ANALYTICS_TABLE_NAME) -> str:
        """Create DynamoDB analytics table if it doesn't exist."""
        dynamodb = self.session.client("dynamodb", region_name=AWS_REGION)
        try:
            dynamodb.describe_table(TableName=table_name)
            logger.info(f"DynamoDB table already exists: {table_name}")
            return table_name
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.info(f"Creating DynamoDB table: {table_name}")
                dynamodb.create_table(
                    TableName=table_name,
                    AttributeDefinitions=[
                        {"AttributeName": "interaction_id", "AttributeType": "S"},
                    ],
                    KeySchema=[
                        {"AttributeName": "interaction_id", "KeyType": "HASH"},
                    ],
                    BillingMode="PAY_PER_REQUEST",
                )
                waiter = dynamodb.get_waiter("table_exists")
                waiter.wait(TableName=table_name)
                logger.info(f"DynamoDB table created: {table_name}")
                return table_name
            raise

    def setup_all_resources(self) -> dict:
        """Set up all required AWS resources (S3 bucket + DynamoDB analytics table)."""
        logger.info("Setting up AWS resources for StudyAgent AI...")
        bucket = self.create_s3_bucket()
        table = self.create_analytics_table()
        return {"s3_bucket": bucket, "analytics_table": table}


# =============================================================================
# Convenience Functions
# =============================================================================
_resource_manager = None


def get_resource_manager() -> AWSResourceManager:
    """Get or create the global resource manager."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = AWSResourceManager()
    return _resource_manager


def upload_document(file_path: str, module: str) -> str:
    """Upload a document to S3."""
    return get_resource_manager().upload_document_to_s3(file_path, module)


def get_document_context(module: str = None, max_chars: int = 50000) -> str:
    """Get all documents as context string."""
    return get_resource_manager().get_all_document_contents(
        module=module, max_chars=max_chars
    )


def list_all_files(bucket_name: str = S3_BUCKET_NAME) -> list:
    """List all documents in S3 with module and filename info (no content loaded)."""
    return get_resource_manager().list_documents(module=None, bucket_name=bucket_name)
