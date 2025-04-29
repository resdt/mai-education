-- Поиск по email (точный поиск)
SET enable_seqscan = OFF;

EXPLAIN ANALYZE
SELECT * FROM users WHERE email = 'JohnParker1@gmail.com';

-- Поиск всех пользователей с ролью 'administrator'
SET enable_seqscan = OFF;

EXPLAIN ANALYZE
SELECT * FROM users WHERE role = 'administrator';

-- Поиск всех пользователей, у кого money_spent больше 5000
SET enable_seqscan = OFF;

EXPLAIN ANALYZE
SELECT * FROM users WHERE money_spent > 5000;