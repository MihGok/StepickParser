class LessonAnalyzer:
    STEP_FILENAME_PREFIX = "step_"
    STEP_FILENAME_SUFFIX = ".json"

    def __init__(self, lesson_dir: str, knowledge_base_dir: str):
        self.lesson_dir = lesson_dir
        self.knowledge_base_dir = knowledge_base_dir
        self.storage = StorageService()
        self.llm_service = GeminiService()
        
        # НОВОЕ: Создаем директорию для временных файлов
        self.temp_base_dir = AppConfig.TEMP_DIR
        os.makedirs(self.temp_base_dir, exist_ok=True)

    def iter_step_files(self) -> Iterator[str]:
        if not os.path.isdir(self.lesson_dir):
            return
        for fname in sorted(os.listdir(self.lesson_dir)):
            if fname.startswith(self.STEP_FILENAME_PREFIX) and fname.endswith(self.STEP_FILENAME_SUFFIX):
                yield os.path.join(self.lesson_dir, fname)

    def _clean_lesson_title(self, dir_name: str) -> str:
        match = re.search(r'^Lesson_\d+_(.+)$', dir_name, re.IGNORECASE)
        clean_name = match.group(1).strip() if match else dir_name.replace('_', ' ').strip()
        return re.sub(r'[<>:"/\\|?*]', '', clean_name).strip()

    def _get_analysis_from_gemini(self, transcript: str, duration: float) -> Optional[VideoAnalysisResult]:
        """Получает таймкоды и 'reason' от Gemini"""
        if not transcript:
            return None
        
        prompt = ANALYZE_TRANSCRIPT_PROMPT_RU.format(
            duration=duration,
            transcript=transcript[:50000]
        )
        
        try:
            return self.llm_service.generate(
                prompt=prompt,
                response_schema=VideoAnalysisResult,
                model_name=AppConfig.GEMINI_MODEL,
                temperature=0.2
            )
        except Exception as e:
            print(f"   [Gemini Error] {e}")
            return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Считает косинусную схожесть двух текстов через TextEncoder (MPNet)"""
        if not text1 or not text2:
            return 0.0
        
        vec1 = Client.get_text_embedding(text1)
        vec2 = Client.get_text_embedding(text2)
        
        if not vec1 or not vec2:
            return 0.0
        
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norm_product == 0:
            return 0.0
            
        return float(dot_product / norm_product)

    def _process_video(self, video_url: str, transcript: str, step_id: int, lesson_name: str) -> List[Dict]:
        """
        ИСПРАВЛЕНО: Скачивание видео с правильным прокси
        """
        temp_dir = os.path.join(self.temp_base_dir, f"frames_{step_id}")
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, "video.mp4")
        final_frames_data = []

        try:
            # ИСПРАВЛЕНИЕ: Используем ProxyConfig для скачивания
            print(f"   [Video] Скачивание с URL: {video_url[:50]}...")
            if not ProxyConfig.download_file(video_url, video_path, use_proxy=True):
                print("   [Video] Не удалось скачать видео")
                return []

            # Проверка что файл скачался
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                print("   [Video] Файл пуст или не создан")
                return []

            # Метаданные видео
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("   [Video] OpenCV не смог открыть видео")
                return []
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frames_count / fps if fps > 0 else 0
            
            if duration == 0:
                print("   [Video] Не удалось определить длительность")
                cap.release()
                return []
            
            print(f"   [Video] Длительность: {duration:.1f}s, FPS: {fps:.1f}")
            
            # Gemini Анализ
            analysis = self._get_analysis_from_gemini(transcript, duration)
            if not analysis or not analysis.timestamps:
                print("   [Video] Gemini не вернул таймкоды")
                cap.release()
                return []

            timestamps_tasks = analysis.timestamps
            print(f"   [Video] Анализ {len(timestamps_tasks)} смысловых блоков...")

            # Цикл по смысловым блокам
            for idx, task in enumerate(timestamps_tasks):
                start_time = task.timestamp
                expected_reason = task.reason
                
                if start_time > duration:
                    print(f"   [Skip] Таймкод {start_time}s вне диапазона видео")
                    continue
                
                print(f"   [{idx+1}/{len(timestamps_tasks)}] t={start_time:.1f}s: {expected_reason[:40]}...")
                
                candidates = []

                # Проверяем 5 кандидатов с интервалом 3 секунды
                for offset in [0, 3, 6, 9, 12]:
                    current_ts = start_time + offset
                    if current_ts > duration:
                        break
                    
                    cap.set(cv2.CAP_PROP_POS_MSEC, current_ts * 1000)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue
                    
                    # Сохраняем кандидат
                    cand_path = os.path.join(temp_dir, f"cand_{int(current_ts)}.jpg")
                    if not cv2.imwrite(cand_path, frame):
                        print(f"      [Warn] Не удалось сохранить {cand_path}")
                        continue
                    
                    # Получаем описание LLaVA
                    llava_desc = Client.get_image_description(cand_path)
                    if not llava_desc:
                        continue
                    
                    # Считаем схожесть
                    score = self._calculate_similarity(expected_reason, llava_desc)
                    print(f"      t={current_ts:.1f}s | Score: {score:.3f}")
                    
                    if score > 0.55:  # Порог снижен с 0.60
                        candidates.append({
                            "score": score,
                            "path": cand_path,
                            "desc": llava_desc,
                            "ts": current_ts
                        })
                
                # Выбираем лучший кадр
                if candidates:
                    best_cand = max(candidates, key=lambda x: x["score"])
                    print(f"   ✓ Выбран кадр t={best_cand['ts']:.1f}s (Score: {best_cand['score']:.3f})")
                    
                    # Загрузка в MinIO
                    s3_key = f"{lesson_name}/step_{step_id}/{int(best_cand['ts'])}.jpg"
                    if self.storage.upload_frame(best_cand["path"], s3_key):
                        final_frames_data.append({
                            "timestamp": best_cand["ts"],
                            "frame_path": s3_key,
                            "description": best_cand["desc"],
                            "validation_score": best_cand["score"],
                            "expected_reason": expected_reason
                        })
                else:
                    print(f"   ✗ Нет подходящих кадров для: '{expected_reason[:40]}...'")

            cap.release()
            print(f"   [Video] Итого выбрано кадров: {len(final_frames_data)}")
            
        except Exception as e:
            print(f"   [Video Error] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Очистка временных файлов
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"   [Cleanup Warning] Не удалось удалить {temp_dir}: {e}")

        return final_frames_data

    def _save_lesson_content(self, all_parsed_steps: List[Dict], lesson_name: str):
        """Сохраняет весь урок в один файл content.txt"""
        lesson_dir = os.path.join(self.knowledge_base_dir, lesson_name)
        os.makedirs(lesson_dir, exist_ok=True)
        filepath = os.path.join(lesson_dir, "content.txt")

        parts = [f"LESSON: {lesson_name}", "="*50]

        for step in all_parsed_steps:
            parts.append(f"\nSTEP ID: {step['step_id']}")
            if step.get('update_date'):
                parts.append(f"UPDATED: {step['update_date']}")
            parts.append("-" * 20)
            
            if step.get("text"):
                parts.append(step["text"])
            
            if step.get("transcript"):
                parts.append("\n[TRANSCRIPT]:")
                parts.append(step["transcript"])

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(parts))
            print(f"   [KB] Сохранен текст урока: {filepath}")
        except Exception as e:
            print(f"   [KB Error] Не удалось сохранить {filepath}: {e}")

    def _save_frames_metadata(self, all_frames: List[Dict], lesson_name: str):
        """Сохраняет метаданные картинок для индексации"""
        if not all_frames:
            return
        
        path = os.path.join(self.knowledge_base_dir, lesson_name, "frames_metadata.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(all_frames, f, ensure_ascii=False, indent=2)
            print(f"   [KB] Сохранено {len(all_frames)} кадров в метаданные")
        except Exception as e:
            print(f"   [KB Error] Ошибка сохранения метаданных: {e}")

    def parse(self) -> List[Dict[str, Any]]:
        """Главный метод парсинга урока"""
        parsed_steps = []
        all_frames_metadata = []
        
        raw_lesson_dir_name = os.path.basename(self.lesson_dir)
        clean_name = self._clean_lesson_title(raw_lesson_dir_name)
        
        print(f"\n[Lesson] Обработка: {clean_name}")

        for step_file in self.iter_step_files():
            try:
                with open(step_file, "r", encoding="utf-8") as f:
                    raw_step = json.load(f)
            except Exception as e:
                print(f"   [Error] Не удалось прочитать {step_file}: {e}")
                continue

            # Базовый парсинг
            parsed = StepAnalyzer.parse_step_dict(raw_step, os.path.basename(step_file))
            if not parsed:
                continue

            # Транскрибация
            if parsed.get("video_url") and not parsed.get("transcript"):
                transcript = Client.transcribe(parsed["video_url"], parsed["step_id"])
                if transcript:
                    parsed["transcript"] = transcript
                    # Кеш обратно в JSON
                    raw_step["_generated_transcript"] = transcript
                    try:
                        with open(step_file, "w", encoding="utf-8") as f:
                            json.dump(raw_step, f, ensure_ascii=False, indent=2)
                    except:
                        pass

            # Обработка видео
            if parsed.get("video_url") and parsed.get("transcript"):
                frames = self._process_video(
                    parsed["video_url"], 
                    parsed["transcript"], 
                    parsed["step_id"], 
                    clean_name
                )
                if frames:
                    for fr in frames:
                        fr['step_id'] = parsed['step_id']
                        fr['lesson_name'] = clean_name
                    all_frames_metadata.extend(frames)

            parsed_steps.append(parsed)

        # Сохранение результатов
        self._save_lesson_content(parsed_steps, clean_name)
        self._save_frames_metadata(all_frames_metadata, clean_name)
        
        return parsed_steps
