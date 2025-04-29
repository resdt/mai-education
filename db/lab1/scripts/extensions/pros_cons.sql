-- ========================================
-- Сравнение pg_trgm и pg_bigm: плюсы и минусы
-- ========================================

-- 📌 Подготовка
SET enable_seqscan = OFF;

-- Поиск подстроки с pg_trgm
EXPLAIN ANALYZE SELECT * FROM users WHERE email ILIKE '%anderson%';

-- Поиск подстроки с bigm
EXPLAIN ANALYZE SELECT * FROM users WHERE email LIKE '%anderson%';

-- 📊 Вывод:
-- pg_trgm:
--   + Поддерживает SIMILARITY, %, <-> и др.
--   + Быстрый поиск по неупорядоченному тексту.
--   – Индексы большие по размеру, медленно строятся.
-- pg_bigm:
--   + Быстрее для длинных запросов (%abc%) и азиатских языков.
--   – Ограничен в возможностях (нет similarity).
--   – Может потребовать GIST вместо GIN.