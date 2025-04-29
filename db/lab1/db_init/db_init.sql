-- Удаляем таблицу users, если существует
DROP TABLE IF EXISTS users;

-- Создаём новую таблицу users
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    role TEXT,
    money_spent NUMERIC,
    category TEXT
);

-- Загружаем данные из CSV
COPY users(first_name, last_name, email, role, money_spent, category)
FROM '/app/users.csv'
DELIMITER ','
CSV HEADER;
