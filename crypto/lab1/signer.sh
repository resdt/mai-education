#!/bin/bash

# Проверка наличия GnuPG и установка при необходимости
if ! command -v gpg &> /dev/null; then
  echo "Установка GnuPG..."
  sudo apt update && sudo apt install -y gnupg
else
  echo "GnuPG уже установлен."
fi

# Подготовка директорий для ключей
mkdir -p signed_keys

# Функция проверки существования ключа
key_exists() {
  gpg --list-keys "$1" &> /dev/null
}

# Создание собственного ключа
read -rp "Введите ваше имя и фамилию на латинице через пробел: " user_name
read -rp "Введите вашу почту: " user_email

if key_exists "$user_email"; then
  echo "Ключ с почтой $user_email уже существует. Пропускаем создание."
else
  gpg --batch --generate-key <<EOF
    Key-Type: RSA
    Key-Length: 4096
    Name-Real: $user_name
    Name-Email: $user_email
    Expire-Date: 365
EOF

  echo "Ключ создан."
fi

export_filename="$(echo "$user_name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_').asc"
gpg --export -a "$user_email" > "$export_filename"
echo "Публичный ключ экспортирован в файл $export_filename"

# Импорт всех ключей из папки src_keys
for friend_key_file in src_keys/*.asc; do
  if [ ! -f "$friend_key_file" ]; then
    echo "Нет файлов для импорта в папке src_keys."
    break
  fi

  gpg --import "$friend_key_file"
done

echo "Доступные ключи после импорта:"
gpg --list-keys --with-colons | awk -F: '/^uid:/ {print $10}'

# Подпись ключей одногруппников
read -rp "Введите почты одногруппников для подписи ключей (через пробел): " -a friend_emails

for friend_email in "${friend_emails[@]}"; do
  if key_exists "$friend_email"; then
    gpg --sign-key "$friend_email"
    signed_filename="signed_${friend_email}_key.asc"
    gpg --export -a "$friend_email" > "signed_keys/$signed_filename"
    echo "Подписанный ключ сохранен в файл signed_keys/$signed_filename"
  else
    echo "Ключ с почтой $friend_email не найден. Пропускаем."
  fi
done
