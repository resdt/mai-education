-- Удаляем B-tree индекс на email
DROP INDEX IF EXISTS idx_users_email;

-- Удаляем B-tree индекс на money_spent
DROP INDEX IF EXISTS idx_users_money_spent;

-- Удаляем GIN индекс на role
DROP INDEX IF EXISTS idx_users_role_gin;

-- Удаляем BRIN индекс на money_spent
DROP INDEX IF EXISTS idx_users_money_spent_brin;
