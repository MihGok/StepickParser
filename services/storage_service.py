import boto3
from botocore.exceptions import ClientError
import cv2
import numpy as np
import io
import os

class StorageService:
    def __init__(self):
        # В продакшене используйте переменные окружения
        self.endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "admin")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "password123")
        self.bucket = "course-frames"
        
        self.s3 = boto3.client('s3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
        self._ensure_bucket()

    def _ensure_bucket(self):
        """Создает бакет, если его нет"""
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except ClientError:
            try:
                self.s3.create_bucket(Bucket=self.bucket)
                print(f"[Storage] Создан бакет: {self.bucket}")
            except Exception as e:
                print(f"[Storage] Ошибка создания бакета: {e}")

    def upload_frame(self, frame_path: str, object_name: str) -> str:
        """
        Загружает файл с диска в S3.
        Args:
            frame_path: Локальный путь к файлу
            object_name: Имя ключа в S3 (например, 'Lesson1/img.jpg')
        Returns:
            object_name если успешно, иначе пустая строка
        """
        try:
            self.s3.upload_file(frame_path, self.bucket, object_name)
            return object_name
        except Exception as e:
            print(f"[Storage] Ошибка загрузки {object_name}: {e}")
            return ""

    def get_presigned_url(self, object_name: str, expiration=3600) -> str:
        """Генерирует временную ссылку на картинку"""
        if not object_name: return ""
        try:
            return self.s3.generate_presigned_url('get_object',
                Params={'Bucket': self.bucket, 'Key': object_name},
                ExpiresIn=expiration
            )
        except ClientError as e:
            print(f"[Storage] Ошибка генерации ссылки: {e}")
            return ""
