-- ========================================
-- ИНТЕГРАЦИЯ pg_trgm и pg_bigm в предметную область (таблица users)
-- ========================================

-- 1. Устанавливаем расширения
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE EXTENSION IF NOT EXISTS pg_bigm;

-- 2. Добавим индексы для полнотекстового поиска по полю email

-- Индекс с использованием trigram (pg_trgm)
CREATE INDEX idx_users_email_trgm ON users USING GIN (email gin_trgm_ops);

-- Индекс с использованием bigram (pg_bigm)
-- Требует изменения метода доступа:
CREATE INDEX idx_users_email_bigm ON users USING GIST (email gist_bigm_ops);
-- или USING gin (email gin_bigm_ops);

-- 3. Примеры поиска
-- Поиск по шаблону
SELECT * FROM users WHERE email ILIKE '%anderson%';

-- Поиск с использованием SIMILARITY (pg_trgm)
SELECT email, similarity (email, 'anderson') AS score
FROM users
WHERE
    email % 'anderson'
ORDER BY score DESC
LIMIT 10;