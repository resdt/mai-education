-- Покажи список индексов для таблицы
SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'users';