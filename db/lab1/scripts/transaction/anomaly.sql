-- ==========================================
-- PostgreSQL Transaction Anomalies Demo
-- ==========================================

-- ⚠️ Для запуска в psql или DBeaver открывайте два отдельных сеанса

-- ===============================================================
-- АНОМАЛИЯ 1: Non-Repeatable Read (неповторяемое чтение)
-- ===============================================================
-- 📌 Описание: Два одинаковых SELECT в одной транзакции возвращают разные данные.
-- Возможна при уровне изоляции: READ COMMITTED (по умолчанию).

-- Сеанс 1
BEGIN ISOLATION LEVEL READ COMMITTED;
SELECT role FROM users WHERE email = 'MatthewAnderson1@gmail.com';
-- не выполняй COMMIT

-- Сеанс 2
BEGIN;
UPDATE users SET role = 'admin' WHERE email = 'MatthewAnderson1@gmail.com';
COMMIT;

-- Сеанс 1
SELECT role FROM users WHERE email = 'MatthewAnderson1@gmail.com';
COMMIT;

-- Ожидаемый эффект: роль изменилась между двумя SELECT в одной транзакции.

-- ===============================================================
-- АНОМАЛИЯ 2: Phantom Read (фантомное чтение)
-- ===============================================================
-- 📌 Описание: Повторный SELECT возвращает новые строки, добавленные в другой транзакции.
-- Возможна при уровне изоляции: READ COMMITTED, REPEATABLE READ (в некоторых случаях).

-- Сеанс 1
BEGIN ISOLATION LEVEL READ COMMITTED;
SELECT COUNT(*) FROM users WHERE money_spent > 9000;
-- не выполняй COMMIT

-- Сеанс 2
BEGIN;
INSERT INTO users (first_name, last_name, email, role, money_spent, category)
VALUES ('Phantom', 'User', 'phantom@example.com', 'user', 9500, 'gold');
COMMIT;

-- Сеанс 1
SELECT COUNT(*) FROM users WHERE money_spent > 9000;
COMMIT;

-- Ожидаемый эффект: количество увеличилось — "появился фантом".

-- ===============================================================
-- АНОМАЛИЯ 3: Lost Update (потерянное обновление)
-- ===============================================================
-- 📌 Описание: Изменения одной транзакции перезаписываются другой, без учёта первого обновления.
-- Возможна при уровне: READ COMMITTED

-- Предподготовка (один раз)
UPDATE users SET money_spent = 1000 WHERE email = 'MatthewAnderson1@gmail.com';

-- Сеанс 1
BEGIN ISOLATION LEVEL READ COMMITTED;
SELECT money_spent FROM users WHERE email = 'MatthewAnderson1@gmail.com';
-- предположим: получено 1000
UPDATE users SET money_spent = 1000 + 200 WHERE email = 'MatthewAnderson1@gmail.com';
-- не коммитим

-- Сеанс 2
BEGIN;
SELECT money_spent FROM users WHERE email = 'MatthewAnderson1@gmail.com';
-- предположим: тоже 1000
UPDATE users SET money_spent = 1000 + 500 WHERE email = 'MatthewAnderson1@gmail.com';
COMMIT;

-- Сеанс 1
COMMIT;

-- Ожидаемый эффект: результат не 1700, а 1200 или 1500 — одно из обновлений теряется.