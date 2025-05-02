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

SELECT money_spent FROM users WHERE email = 'BarbaraMorgan1@gmail.com';
-- 🔁 Ждём, не коммитим

-- Session 2
BEGIN;

UPDATE users
SET money_spent = money_spent + 100
WHERE email = 'BarbaraMorgan1@gmail.com';

COMMIT;

-- Session 1
SELECT money_spent FROM users WHERE email = 'BarbaraMorgan1@gmail.com' LIMIT 100;
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

SELECT money_spent FROM users WHERE email = 'BarbaraMorgan1@gmail.com';
-- 🔁 Ждём, не коммитим

-- Session 2
BEGIN;

UPDATE users
SET money_spent = money_spent + 100
WHERE email = 'BarbaraMorgan1@gmail.com';

COMMIT;

-- Session 1
SELECT money_spent FROM users WHERE email = 'BarbaraMorgan1@gmail.com';
COMMIT;

-- ✅ Ожидаемый результат:
-- Второй SELECT вернёт старое значение — как будто обновления не было.
-- PostgreSQL использует снимок данных, полученный в начале транзакции.

-- =====================================================================
-- ТЕСТ 3: Поведение при SERIALIZABLE
-- =====================================================================
-- 📌 Описание:
--   SERIALIZABLE имитирует параллельную работу, как если бы транзакции
--   выполнялись последовательно. Попытка обновить одну и ту же строку
--   может привести к ошибке сериализации (serialization failure).
--   Это предотвращает phantom reads, non-repeatable reads и т.д.

-- Session 1
BEGIN ISOLATION LEVEL SERIALIZABLE;

SELECT money_spent FROM users WHERE email = 'BarbaraMorgan1@gmail.com';
-- 🔁 Ждём, не коммитим

-- Session 2
BEGIN ISOLATION LEVEL SERIALIZABLE;

UPDATE users
SET money_spent = money_spent + 100
WHERE email = 'BarbaraMorgan1@gmail.com';

COMMIT;

-- Session 1
UPDATE users
SET money_spent = money_spent + 50
WHERE email = 'BarbaraMorgan1@gmail.com';

-- ⚠️ Возможен результат:
-- ERROR: could not serialize access due to concurrent update
--        (или аналогичное сообщение)
-- ⛔ PostgreSQL откатит транзакцию — потребуется повторить.

COMMIT;

-- ✅ Ожидаемое поведение:
-- Один из сеансов завершится ошибкой сериализации.
-- Нужно будет повторить транзакцию для сохранения согласованности данных.