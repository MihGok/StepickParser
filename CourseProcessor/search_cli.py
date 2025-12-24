import argparse
from CourseProcessor.indexing.qdrant_indexer import QdrantKnowledgeBaseIndexer
from services.storage_service import StorageService

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('query', type=str, help='Поисковый запрос')
    parser.add_argument('--kb', required=True, help='Путь к папке базы знаний')
    parser.add_argument('--mode', choices=['text', 'image', 'ctx'], default='text')
    args = parser.parse_args()

    indexer = QdrantKnowledgeBaseIndexer(knowledge_base_dir=args.kb)
    storage = StorageService()

    if args.mode == 'text':
        results = indexer.search_text(args.query)
        for i, res in enumerate(results, 1):
            print(f"\n{i}. Урок: {res['lesson_name']} (Чанк {res['chunk_index']}/{res['total_chunks']})")
            print(f"   Score: {res['score']:.4f}")
            print(f"   Текст: {res['text'][:200]}...")

    elif args.mode == 'image':
        results = indexer.search_images(args.query)
        for i, res in enumerate(results, 1):
            url = storage.get_presigned_url(res['s3_key'])
            print(f"\n{i}. [IMAGE] {res['lesson_name']} ({res['timestamp']:.1f}s)")
            print(f"   Score: {res['score']:.4f}")
            print(f"   Описание: {res['description'][:150]}...")
            print(f"   Ссылка: {url}")

    elif args.mode == 'ctx':
        # Пример: query это точное название урока
        chunks = indexer.get_lesson_context(args.query)
        print(f"\nВосстановлен контекст урока '{args.query}':")
        print(f"Всего чанков: {len(chunks)}")
        full_text = " ".join([c['text'] for c in chunks])
        print(f"Полный текст (первые 500 симв): {full_text[:500]}...")

if __name__ == "__main__":
    main()
