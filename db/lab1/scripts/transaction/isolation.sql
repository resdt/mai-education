-- ==========================================
-- PostgreSQL Isolation Levels Demonstration
-- ==========================================

-- =====================================================================
-- ТЕСТ 1: Поведение при READ COMMITTED (по умолчанию)
-- =====================================================================
-- 📌 Описание:
--   В первом сеансе выполняется SELECT, затем параллельное обновление во втором сеансе.
--   При повторном SELECT видно новое значение.
--   Это демонстрирует "неповторяемое чтение" (non-repeatable read).

-- Session 1
BEGIN ISOLATION LEVEL READ COMMITTED;

SELECT money_spent FROM users WHERE email = 'MatthewAnderson1@gmail.com';
-- 🔁 Ждём, не коммитим

-- Session 2
BEGIN;

UPDATE users
SET money_spent = money_spent + 100
WHERE email = 'MatthewAnderson1@gmail.com';

COMMIT;

-- Session 1
SELECT money_spent FROM users WHERE email = 'MatthewAnderson1@gmail.com';
COMMIT;

-- ✅ Ожидаемый результат:
-- Второй SELECT покажет обновлённое значение. Данные "поменялись внутри транзакции".

-- =====================================================================
-- ТЕСТ 2: Поведение при REPEATABLE READ
-- =====================================================================
-- 📌 Описание:
--   При REPEATABLE READ второй SELECT в первом окне возвращает те же данные,
--   даже если во втором окне произошёл COMMIT.
--   Это предотвращает non-repeatable read.

-- Session 1
BEGIN ISOLATION LEVEL REPEATABLE READ;

SELECT money_spent FROM users WHERE email = 'MatthewAnderson1@gmail.com';
-- 🔁 Ждём, не коммитим

-- Session 2
BEGIN;

UPDATE users
SET money_spent = money_spent + 100
WHERE email = 'MatthewAnderson1@gmail.com';

COMMIT;

-- Session 1
SELECT money_spent FROM users WHERE email = 'MatthewAnderson1@gmail.com';
COMMIT;

-- ✅ Ожидаемый результат:
-- Второй SELECT вернёт старое значение — как будто обновления не было.
-- PostgreSQL использует снимок данных, полученный в начале транзакции.