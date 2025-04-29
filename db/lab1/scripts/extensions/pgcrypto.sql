-- ========================================
-- Работа с pgcrypto: хэширование и шифрование
-- ========================================

-- 1. Установить расширение
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 2. Добавить колонку для хранения хэша пароля
ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT;

-- 3. Сгенерировать хэш пароля
UPDATE users
SET
    password_hash = crypt (
        'super_secure_password',
        gen_salt ('bf')
    )
WHERE
    email = 'MatthewAnderson1@gmail.com';

-- 4. Проверка пароля
SELECT email
FROM users
WHERE
    email = 'MatthewAnderson1@gmail.com'
    AND password_hash = crypt (
        'super_secure_password',
        password_hash
    );

-- 📌 Безопасность:
-- - Пароли не хранятся в открытом виде.
-- - Соль генерируется отдельно для каждого пользователя.
-- - Устойчиво к атакам со стороны БД-админа.

-- Можно также использовать: digest('text', 'sha256') для вычисления хэша без соли.