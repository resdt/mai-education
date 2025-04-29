-- B-tree индекс на email (для точного поиска по email)
CREATE INDEX idx_users_email ON users (email);

-- B-tree индекс на money_spent (для поиска по числовым диапазонам)
CREATE INDEX idx_users_money_spent ON users (money_spent);

-- GIN индекс на role с помощью триграмм (для быстрого поиска по тексту)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE INDEX idx_users_role_gin ON users USING GIN (ROLE gin_trgm_ops);

-- BRIN индекс на money_spent (если таблица огромная и money_spent распределено равномерно)
CREATE INDEX idx_users_money_spent_brin ON users USING BRIN (money_spent);