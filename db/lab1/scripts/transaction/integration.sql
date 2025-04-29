-- Перевод денег между пользователями
BEGIN;

UPDATE users
SET
    money_spent = money_spent - 100
WHERE
    email = 'UserA@example.com';

UPDATE users
SET
    money_spent = money_spent + 100
WHERE
    email = 'UserB@example.com';

COMMIT;

-- Понижение уровня привилегий
BEGIN;

UPDATE users
SET ROLE = 'user'
WHERE
    ROLE = 'administrator'
    AND money_spent < 3000;

COMMIT;

-- Массовое обновление по категории
BEGIN;

UPDATE users SET category = 'platinum' WHERE money_spent > 9500;

COMMIT;